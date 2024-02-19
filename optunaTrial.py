import math
import pathlib
import optuna
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from setuphandler import setup_datamodule, setup_trainingmode, setup_trainingparameter
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything


# Load Configs
with open("config.yml", "r") as config_file:
     config = yaml.safe_load(config_file)

pretrained_model_path = config["pretrained_model_path"]

dataset_size = config["dataset_size"]
img_size_param = config["img_size_param"]

alt_params = config["alt_params"]

every_n_samples_loss = config["every_n_samples_loss"]

optuna_suggest = config["optuna_suggest"]


project_name = config["project_name"]


dataset_name = config["dataset_name"]
animesketchcolorizationpair_train = pathlib.Path(config["animesketchcolorizationpair_train"])
animesketchcolorizationpair_test = pathlib.Path(config["animesketchcolorizationpair_test"])


training_mode = config["training_mode"]
test_params = config["test_params"]


def objective(trial):
    wandb.finish()
    seed_everything(7, True)
    # Define the hyperparameters to optimize.
    model_hparams, data_transforms = setup_trainingparameter(trial, img_size_param, pretrained_model_path,
                                                             optuna_suggest=optuna_suggest, alt_params=alt_params,
                                                             batch_size_suggest_higher_bound=5)

    dm = setup_datamodule(dataset_name, animesketchcolorizationpair_train, animesketchcolorizationpair_test,
                          model_hparams["batch_size"], data_transforms)

    model = setup_trainingmode(training_mode, model_hparams, pretrained_model_path, every_n_samples_loss)

    # Create a PyTorch Lightning trainer with the logger.
    wandb_logger = WandbLogger(name=f"trial-{trial.number}", project=project_name)

    if test_params["test_mode"]:
        log_interval = 5
        limit_train_batches_param = test_params["limit_train_batches"]
        limit_val_batches_param = test_params["limit_val_batches"]
        limit_test_batches_param = test_params["limit_test_batches"]
    else:
        log_interval = math.floor(math.ceil(dataset_size * 0.9 / alt_params["batch_size"]) * 0.1)
        limit_train_batches_param = None
        limit_val_batches_param = None
        limit_test_batches_param = None

    if training_mode in ["DDPM-Finetuning-load_from_checkpoint", "DDPM-Finetuning"]:
        filename_param = "checkpoint_{epoch:02d}-{train_loss:.4f}-{step:02d}"
        every_n_train_steps_param = math.floor(math.ceil(dataset_size * 0.9 / alt_params["batch_size"]) * 0.1)
        every_n_epochs_param = None
    else:
            filename_param = "checkpoint_{epoch:02d}-{val_loss:.4f}-{step:02d}"
            every_n_epochs_param = 1
            every_n_train_steps_param = None

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./models/{project_name}/checkpoints-trial-{trial.number}",
        filename=filename_param,
        every_n_epochs=every_n_epochs_param,
        every_n_train_steps=every_n_train_steps_param,
        auto_insert_metric_name=True,
        save_on_train_epoch_end=True,
    )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=model_hparams["accumulate_grad_batches"],
        max_epochs=4,
        limit_train_batches=limit_train_batches_param,
        limit_val_batches=limit_val_batches_param,
        limit_test_batches=limit_test_batches_param,
        check_val_every_n_epoch=1,
        accelerator="auto",
        precision="16",
        log_every_n_steps=log_interval,
        benchmark=True,
    )
    torch.set_float32_matmul_precision("high")

    if training_mode in ["DDPM-load_from_checkpoint", "DDPM-Finetuning-load_from_checkpoint"]:
        checkpoint_load_path = pretrained_model_path
    else:
        checkpoint_load_path = None

    trainer.fit(model=model, datamodule=dm, ckpt_path=checkpoint_load_path)
    val_loss = trainer.callback_metrics["val_loss"]
    trainer.save_checkpoint(f"./models/{project_name}/trial-{trial.number}.ckpt")
    trainer.test(model=model, datamodule=dm)
    return val_loss


if __name__ == "__main__":
    study_name = project_name
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///./databank/{study_name}.db",
        load_if_exists=True,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=1)
