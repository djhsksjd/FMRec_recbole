"""
RecBole training entrypoint without `recbole.quick_start`.

`recbole.quick_start` pulls in optional dependencies (e.g. `ray`) that may not
be installed in this environment, so we wire up the standard RecBole pipeline
directly: Config -> Dataset -> DataLoader -> Trainer.
"""

import argparse
import sys
import numpy as np

# RecBole versions that predate NumPy 2.0 may reference removed aliases.
# Patch the minimal set needed before importing RecBole.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "int_"):
    np.int_ = np.int64  # type: ignore[attr-defined]
if not hasattr(np, "bool_"):
    np.bool_ = np.bool8  # type: ignore[attr-defined]
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128  # type: ignore[attr-defined]
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_  # type: ignore[attr-defined]

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed

from model.recbole_hr_metric import register_hr_metric
from model.fmrec import FMRecRecBole
from model.fmrec_trainer import FMRecTrainer

# Register `hr` before any Evaluator/Collector is built (same math as RecBole Hit, keys hr@{k}).
register_hr_metric()



def main() -> None:
    # Parse custom arguments before RecBole's Config
    parser = argparse.ArgumentParser(description="FMRec training and inference", add_help=False)
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train',
                        help='Mode: train to train the model, inference to only evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for inference mode (optional)')
    
    # Extract our custom args, keep the rest for RecBole
    args, remaining_argv = parser.parse_known_args()
    
    # Temporarily modify sys.argv to remove our custom arguments
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]] + remaining_argv

    config = Config(model=FMRecRecBole, dataset=None, config_file_list=["configs.yaml"])
    
    # Restore original sys.argv
    sys.argv = original_argv

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = FMRecRecBole(config, train_data.dataset).to(config["device"])
    trainer = FMRecTrainer(config, model)

    if args.mode == 'train':
        # 训练模式
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])
        trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])
    else:
        # 推理模式 - 只进行评估，不训练
        trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])




if __name__ == "__main__":
    main()