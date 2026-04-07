import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_normal_
from typing import Optional, Tuple

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class LayerNorm(nn.Module):
    """Keep the same TF-style LayerNorm as the original code."""

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.constant_(self.w_1.bias, 0)
        nn.init.xavier_normal_(self.w_2.weight)
        nn.init.constant_(self.w_2.bias, 0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (
            1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3)))
        )
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int, hidden_size: int, dropout: float):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by heads ({heads}).")
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        q, k, v = [
            l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
            for l, x in zip(self.linear_layers, (q, k, v))
        ]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            # mask: [B, L] -> broadcast to [B, heads, L, L]
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1, 1, 1, corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)

        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, attn_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class TransformerRep(nn.Module):
    def __init__(self, hidden_size: int, num_blocks: int, dropout: float, last: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = 4
        self.dropout = dropout
        self.n_blocks = num_blocks
        self.last = last
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)]
        )

    def forward(self, hidden: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encode = hidden
        for i, transformer in enumerate(self.transformer_blocks, start=1):
            hidden = transformer(hidden, c, mask)
            if i == (self.n_blocks - self.last):
                encode = hidden
        return hidden, encode


class FMXStart(nn.Module):
    def __init__(self, hidden_size: int, *, item_num: int, num_blocks: int, dropout: float, last: int, lambda_uncertainty: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))
        self.fuse_linear = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.att = TransformerRep(hidden_size=self.hidden_size, num_blocks=num_blocks, dropout=dropout, last=last)

        self.lambda_uncertainty = lambda_uncertainty
        self.dropout = nn.Dropout(dropout)
        self.norm_fm_rep = LayerNorm(self.hidden_size)

        self.item_num = item_num
        self.out_dims = [512, 2048]
        self.act_func = "tanh"

        out_dims_temp = [self.hidden_size] + self.out_dims + [self.item_num]
        decoder_modules: list[nn.Module] = []
        for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
            decoder_modules.append(nn.Linear(d_in, d_out))
            if self.act_func == "relu":
                decoder_modules.append(nn.ReLU())
            elif self.act_func == "sigmoid":
                decoder_modules.append(nn.Sigmoid())
            elif self.act_func == "tanh":
                decoder_modules.append(nn.Tanh())
            elif self.act_func == "leaky_relu":
                decoder_modules.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unsupported act_func={self.act_func}")
        decoder_modules.pop()
        self.decoder = nn.Sequential(*decoder_modules)
        self.decoder.apply(self._xavier_normal_initialization)

    @staticmethod
    def _xavier_normal_initialization(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(
        self, rep_item: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, mask_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))

        lambda_uncertainty = torch.normal(
            mean=torch.full(rep_item.shape, self.lambda_uncertainty, device=x_t.device),
            std=torch.full(rep_item.shape, self.lambda_uncertainty, device=x_t.device),
        )

        rep_item_new = rep_item + (lambda_uncertainty * (x_t + emb_t).unsqueeze(1))
        condition_cross = rep_item

        rep_fm, encode = self.att(rep_item_new, condition_cross, mask_seq)
        rep_fm = self.norm_fm_rep(self.dropout(rep_fm))

        out = rep_fm[:, -1, :]
        encoded = encode[:, -1, :]
        decode = self.decoder(encoded)  # [B, n_items]
        return out, decode


class FMRec(nn.Module):
    """Your original FMRec core, kept intact (Euler sampler; SciPy not required)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        item_num: int,
        num_blocks: int,
        dropout: float,
        last: int,
        lambda_uncertainty: float,
        eps: float,
        sample_N: int,
        eps_reverse: float,
        m_logNorm: float,
        s_logNorm: float,
        s_modsamp: float,
        sampling_method: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.xstart_model = FMXStart(
            hidden_size=hidden_size,
            item_num=item_num,
            num_blocks=num_blocks,
            dropout=dropout,
            last=last,
            lambda_uncertainty=lambda_uncertainty,
        )
        self.eps = eps
        self.sample_N = sample_N
        self.eps_reverse = eps_reverse
        self.m_logNorm = m_logNorm
        self.s_logNorm = s_logNorm
        self.s_modsamp = s_modsamp
        self.sampling_method = sampling_method

    @property
    def T(self) -> float:
        return 1.0

    @staticmethod
    def a_t_fn(t: torch.Tensor) -> torch.Tensor:
        return t

    @staticmethod
    def b_t_fn(t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def q_sample_rf(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        z0: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Linear blend x_t = a_t * x_start + b_t * z0; mask==0 keeps x_start (no noise on padding)."""
        assert z0.shape == x_start.shape
        a_t = self.a_t_fn(t)
        b_t = self.b_t_fn(t)
        x_t = a_t * x_start + b_t * z0
        if mask is None:
            return x_t
        mask = torch.broadcast_to(mask.unsqueeze(-1), x_start.shape)
        return torch.where(mask == 0, x_start, x_t)

    @staticmethod
    def logit_normal_sampling_torch(m: float, s: float, batch_size: int, device: torch.device) -> torch.Tensor:
        u_samples = torch.normal(mean=m, std=s, size=(batch_size,), device=device)
        return 1 / (1 + torch.exp(-u_samples))

    @staticmethod
    def mode_sample_timestep(batch_size: int, s: float, device: torch.device) -> torch.Tensor:
        u = torch.rand(batch_size, device=device)
        correction_term = s * (torch.cos((torch.pi / 2) * u) ** 2 - 1 + u)
        return 1 - u - correction_term

    @staticmethod
    def cosmap_sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
        u = torch.rand(batch_size, device=device)
        return 1 - 1 / (torch.tan((torch.pi / 2) * u) + 1)

    def euler_sampler(self, item_rep: torch.Tensor, mask_seq: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            device = next(self.xstart_model.parameters()).device
            x = z0.to(device)

            dt = 1.0 / self.sample_N
            eps = self.eps_reverse
            extra = (1 / self.eps_reverse) - 1

            for i in range(self.sample_N):
                num_t = i / self.sample_N * (self.T - eps) + eps
                t = torch.ones(x.shape[0], device=device) * num_t
                pred, _ = self.xstart_model(item_rep, x, t * extra, mask_seq)
                v = pred - z0  # rectified flow
                x = x.detach().clone() + v * dt

            return x

    def reverse_p_sample_rf(self, item_rep: torch.Tensor, z0: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
        return self.euler_sampler(item_rep, mask_seq, z0)

    def forward(
        self, item_rep: torch.Tensor, item_tag: torch.Tensor, mask_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(item_tag)
        z0 = noise
        batch_size = item_tag.shape[0]

        if self.sampling_method == "mode":
            t_rf = self.mode_sample_timestep(batch_size, self.s_modsamp, item_tag.device) * (self.T - self.eps) + self.eps
        elif self.sampling_method == "uniform":
            t_rf = torch.rand(batch_size, device=item_tag.device) * (self.T - self.eps) + self.eps
        elif self.sampling_method == "logit_normal":
            t_rf = self.logit_normal_sampling_torch(self.m_logNorm, self.s_logNorm, batch_size, item_tag.device) * (self.T - self.eps) + self.eps
        elif self.sampling_method == "cosmap":
            t_rf = self.cosmap_sample_timesteps(batch_size, item_tag.device) * (self.T - self.eps) + self.eps
        else:
            raise ValueError(f"Unsupported sampling_method={self.sampling_method}")

        t_rf_expand = t_rf.view(-1, 1).repeat(1, item_tag.shape[1])
        x_t = self.q_sample_rf(item_tag, t_rf_expand, z0=z0)
        extra = (1 / self.eps) - 1
        x_0, decode_out = self.xstart_model(item_rep, x_t, t_rf * extra, mask_seq)
        return x_0, decode_out, t_rf_expand, t_rf, z0


class FMRecRecBole(SequentialRecommender):
    """RecBole wrapper around the original FMRec method."""

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.num_blocks = config["num_blocks"]
        self.dropout = config["dropout"]
        self.emb_dropout = config["emb_dropout"]
        self.lambda_uncertainty = config["lambda_uncertainty"]
        self.eps = config["eps"]
        self.sample_N = config["sample_N"]
        self.eps_reverse = config["eps_reverse"]
        self.m_logNorm = config["m_logNorm"]
        self.s_logNorm = config["s_logNorm"]
        self.s_modsamp = config["s_modsamp"]
        self.sampling_method = config["sampling_method"]
        self.mask_ratio = config["mask_ratio"]
        self.loss_alpha = config["Loss_Alpha"]
        self.loss_beta = config["Loss_Beta"]
        self.last = config["last"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.layer_norm = LayerNorm(self.hidden_size)
        self.drop = nn.Dropout(self.dropout)
        self.embed_drop = nn.Dropout(self.emb_dropout)

        self.fm_core = FMRec(
            hidden_size=self.hidden_size,
            item_num=self.n_items,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            last=self.last,
            lambda_uncertainty=self.lambda_uncertainty,
            eps=self.eps,
            sample_N=self.sample_N,
            eps_reverse=self.eps_reverse,
            m_logNorm=self.m_logNorm,
            s_logNorm=self.s_logNorm,
            s_modsamp=self.s_modsamp,
            sampling_method=self.sampling_method,
        )

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            # TF-style LayerNorm in this file is nn.Module, not nn.LayerNorm — must init explicitly.
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _encode_item_seq(self, item_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # item_seq: [B, L]
        mask_seq = (item_seq > 0).float()  # [B, L]
        item_emb = self.item_embedding(item_seq)  # [B, L, H]
        item_emb = self.embed_drop(item_emb)

        position_ids = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0).expand(item_seq.size(0), -1)
        item_emb = item_emb + self.position_embedding(position_ids)
        item_emb = self.layer_norm(item_emb)
        item_emb = self.drop(item_emb)
        return item_emb, mask_seq

    def _switch_matrix(self, sequence: torch.Tensor) -> torch.Tensor:
        """Vectorized version of `switch_Matrix`: build multi-hot [B, n_items] from [B, L] item ids."""
        device = sequence.device
        batch_size, seq_len = sequence.size()
        mat = torch.zeros(batch_size, self.n_items, device=device)

        mask = sequence > 0
        if mask.any():
            rows = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, seq_len)[mask]
            cols = sequence[mask]
            mat[rows, cols] = 1.0
        return mat

    def _balanced_mse_loss(self, target: torch.Tensor, output: torch.Tensor, mask_ratio: float = 1.0) -> torch.Tensor:
        # target/output: [B, n_items]
        target = target.float()
        output = output.float()

        num_ones = int(torch.sum(target == 1).item())
        num_zeros = int(torch.sum(target == 0).item())
        num_selected_zeros = int(min(num_zeros, num_ones * mask_ratio))

        zero_positions = (target == 0).nonzero(as_tuple=True)
        one_positions = (target == 1).nonzero(as_tuple=True)

        mask = torch.zeros_like(target)
        if num_selected_zeros > 0:
            zero_rows, zero_cols = zero_positions
            selected_zero_indices = torch.randint(0, num_zeros, (num_selected_zeros,), device=target.device)
            mask[(zero_rows[selected_zero_indices], zero_cols[selected_zero_indices])] = 1.0
        if num_ones > 0:
            mask[one_positions] = 1.0

        masked_target = target * mask
        masked_output = output * mask
        return self.loss_mse(masked_output, masked_target)

    def _forward_train(self, item_seq: torch.Tensor, pos_item: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item_emb, mask_seq = self._encode_item_seq(item_seq)
        tag_emb = self.item_embedding(pos_item)  # [B, H]
        rep_fm, decode_out, _, _, _ = self.fm_core(item_emb, tag_emb, mask_seq)  # rep_fm: [B, H], decode_out: [B, n_items]
        return rep_fm, decode_out, tag_emb

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        # RecBole will call this in evaluation/prediction. We follow your original inference branch: reverse sampling.
        item_emb, mask_seq = self._encode_item_seq(item_seq)
        z0 = torch.randn_like(item_emb[:, -1, :])
        rep_fm = self.fm_core.reverse_p_sample_rf(item_emb, z0, mask_seq)  # [B, H]
        return rep_fm

    def calculate_loss(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        pos_item = interaction[self.POS_ITEM_ID]
        if pos_item.dim() > 1:
            pos_item = pos_item.squeeze(-1)

        rep_fm, decode_out, tag_emb = self._forward_train(item_seq, pos_item)

        # CE over all items (your `loss_fm_ce`)
        scores = torch.matmul(rep_fm, self.item_embedding.weight.t())
        loss_fm_ce = self.loss_ce(scores, pos_item)

        # MSE(rep_fm, tag_emb) (your `loss_FM_mse`)
        loss_fm_mse = self.loss_mse(rep_fm, tag_emb)

        # balanced mse over seq multi-hot vs decode_out (your `balanced_mse_loss`)
        seq_matrix = self._switch_matrix(item_seq)  # [B, n_items]
        loss_decode = self._balanced_mse_loss(seq_matrix, decode_out, self.mask_ratio)

        return loss_fm_mse + self.loss_alpha * loss_fm_ce + self.loss_beta * loss_decode

    def predict(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        rep_fm = self.forward(item_seq, interaction[self.ITEM_SEQ_LEN])
        test_emb = self.item_embedding(test_item)
        return torch.sum(rep_fm * test_emb, dim=-1)

    def full_sort_predict(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        rep_fm = self.forward(item_seq, interaction[self.ITEM_SEQ_LEN])
        return torch.matmul(rep_fm, self.item_embedding.weight.t())