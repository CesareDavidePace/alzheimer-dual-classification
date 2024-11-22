import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torchvision
from models.singleClassifier import SingleInputClassifier
from models.dualClassifier import DualInputClassifier
from data_module import AlzheimerDataModule
from utils.random_seed import set_random_seed
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import argparse
import yaml
import sklearn  
import wandb
import statistics


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    # Set random seed for reproducibility
    set_random_seed(cfg.seed)

    # Print library versions and available GPUs
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")
    print(f"TorchVision version: {torchvision.__version__}\n")
    if torch.cuda.is_available():
        print(f"GPU: {cfg.training.gpus}")  
    else:
        print("No GPU available!")

    # Initialize metrics for overall evaluation
    overall_metrics = {
        "accuracy": [],
        "f1_score": [],
        "sensitivity": [],
        "specificity": [],
        "mcc": []
    }

    # Run experiments for all folds
    for fold in range(1, 6):
        print("\n\033[92m" + f"Running experiment for Fold{fold}..." + "\033[0m")

        if cfg.save_results == True:
            # Setup wandb logger for this fold
            wandb_logger = WandbLogger(
                project="alzheimer-classification",
                entity="krahim04",
                name=f"fold_{fold}",
                group="5-fold CV",
                log_model="all",
            )

            # Log hyperparameters
            wandb_logger.log_hyperparams(cfg)

        # Initialize the data module with the current fold
        data_module = AlzheimerDataModule(
            root_dir=cfg.data.root_dir,
            yaml_path=cfg.data.split_division_path,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            fold=f"Fold{fold}", 
        )

        # Select the appropriate model based on cfg.model.mode
        if cfg.model.mode == "single":
            model = SingleInputClassifier(
                backbone=cfg.model.backbone,
                num_classes=cfg.model.num_classes,
                pretrained=cfg.model.pretrained,
                learning_rate=cfg.model.learning_rate,
            )
        elif cfg.model.mode == "dual":
            model = DualInputClassifier(
                rgb_backbone=cfg.model.backbone,
                single_channel_backbone=cfg.model.backbone,
                num_classes=cfg.model.num_classes,
                pretrained=cfg.model.pretrained,
                learning_rate=cfg.model.learning_rate,
            )
        else:
            raise ValueError(f"Invalid model mode: {cfg.model.mode}. Choose 'single' or 'dual'.")

        # Checkpoint callback
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=cfg.training.monitor_metric,
            mode=cfg.training.monitor_mode,
            save_top_k=cfg.training.save_top_k,
            filename=f"fold{fold}_best_model",
        )

        # Early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor_metric,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            verbose=False,
        )

        # Initialize the trainer
        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            callbacks=[checkpoint_callback, early_stopping_callback],
            accelerator="gpu",
            devices=[int(gpu) for gpu in cfg.training.gpus.split(",")],
            log_every_n_steps=cfg.training.log_every_n_steps,
            enable_model_summary=False,
            logger=wandb_logger if cfg.save_results == True else None,
        )

        # Train the model
        trainer.fit(model, datamodule=data_module)

        # Load the best model
        if cfg.model.mode == "single":
            model = SingleInputClassifier.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,
                num_classes=cfg.model.num_classes,
            )
        elif cfg.model.mode == "dual":
            model = DualInputClassifier.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,
                num_classes=cfg.model.num_classes,
            )

        # Test the model
        test_metrics = trainer.test(model, datamodule=data_module)

        overall_metrics["accuracy"].append(test_metrics[0]["test_accuracy"])
        overall_metrics["f1_score"].append(test_metrics[0]["test_f1"])
        overall_metrics["sensitivity"].append(test_metrics[0]["test_sensitivity"])
        overall_metrics["specificity"].append(test_metrics[0]["test_specificity"])
        overall_metrics["mcc"].append(test_metrics[0]["test_mcc"])

        # Finish the wandb run to avoid conflicts
        if cfg.save_results == True:
            wandb.finish()
        
    # Calculate overall metrics
    mean_accuracy = sum(overall_metrics["accuracy"]) / len(overall_metrics["accuracy"])
    mean_f1_score = sum(overall_metrics["f1_score"]) / len(overall_metrics["f1_score"])
    mean_sensitivity = sum(overall_metrics["sensitivity"]) / len(overall_metrics["sensitivity"])
    mean_specificity = sum(overall_metrics["specificity"]) / len(overall_metrics["specificity"])
    mean_mcc = sum(overall_metrics["mcc"]) / len(overall_metrics["mcc"])

    std_accuracy = statistics.stdev(overall_metrics["accuracy"])
    std_f1_score = statistics.stdev(overall_metrics["f1_score"])
    std_sensitivity = statistics.stdev(overall_metrics["sensitivity"])
    std_specificity = statistics.stdev(overall_metrics["specificity"])
    std_mcc = statistics.stdev(overall_metrics["mcc"])

    print("\n\033[92m" + "Overall Metrics" + "\033[0m")
    print(f"Overall Accuracy: {mean_accuracy:.4f} ({std_accuracy:.4f})")
    print(f"Overall F1 Score: {mean_f1_score:.4f} ({std_f1_score:.4f})")
    print(f"Overall Sensitivity: {mean_sensitivity:.4f} ({std_sensitivity:.4f})")
    print(f"Overall Specificity: {mean_specificity:.4f} ({std_specificity:.4f})")
    print(f"Overall MCC: {mean_mcc:.4f} ({std_mcc:.4f})")


if __name__ == "__main__":
    main()