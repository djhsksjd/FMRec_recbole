"""RecBole Trainer with StepLR, aligned with src/trainer.py (per-epoch scheduler step)."""

from __future__ import annotations

import torch.optim as optim
from recbole.trainer import Trainer


class FMRecTrainer(Trainer):
    """Same as RecBole Trainer but applies torch.optim.lr_scheduler.StepLR each epoch.

    Matches src/main.py: decay_step (step_size), gamma; lr_scheduler.step() every epoch
    after training, like model_train() in src/trainer.py.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        # RecBole Config has no .get(); optional keys live in final_config_dict.
        fcd = config.final_config_dict
        decay_step = int(fcd.get("decay_step", 0) or 0)
        gamma = float(fcd.get("gamma", 0.1))
        if decay_step > 0:
            self._lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=decay_step, gamma=gamma
            )
        else:
            self._lr_scheduler = None

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        out = super()._train_epoch(
            train_data, epoch_idx, loss_func=loss_func, show_progress=show_progress
        )
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        return out
