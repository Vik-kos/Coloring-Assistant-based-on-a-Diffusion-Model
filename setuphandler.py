from DataModule import SketchAndColorDataModule
from DiffusionModel import DDPM_Model, DDPM_Model_finetune
import torchvision.transforms as transforms


def setup_datatransforms(img_size):
    return {
        'colored': transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),
        'grayscale': transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),
        'reference': transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    }


def setup_datamodule(dataset_name, train_path, test_path, batch_size, data_transforms):
    if dataset_name == "AnimeSketchColorizationPair":
        return SketchAndColorDataModule(imagefolder_train=train_path,
                                        imagefolder_test=test_path, extension="png",
                                        batch_size=batch_size,
                                        num_workers=8, transform=data_transforms)
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}.")


def setup_trainingmode(training_mode, model_hparams, pretrained_model_path, every_n_samples_loss):
    if training_mode == "DDPM":
        return DDPM_Model(conf=model_hparams, time_dim=model_hparams["img_size"])

    elif training_mode == "DDPM-Finetuning":
        if model_hparams["sampling_steps"] % every_n_samples_loss != 0:
            raise ValueError(
                "every_n_samples_loss needs to be sampling_steps % every_n_samples_loss == 0."
            )
        return DDPM_Model_finetune(conf=model_hparams, every_n_samples_loss=every_n_samples_loss,
                                   time_dim=model_hparams["img_size"])

    elif training_mode == "DDPM-load_from_checkpoint":
        return DDPM_Model.load_from_checkpoint(pretrained_model_path)

    elif training_mode == "DDPM-Finetuning-load_from_checkpoint":
        return DDPM_Model_finetune.load_from_checkpoint(pretrained_model_path)

    else:
        raise ValueError(f"Unknown training_mode {training_mode}.")


def setuphelper_optuna_int(trial, optuna_suggest, variable_name, bounds, alt_params):
    if optuna_suggest[f"{variable_name}_suggest"]:
        return trial.suggest_int(variable_name, bounds[0], bounds[1])
    else:
        return alt_params[variable_name]


def setup_trainingparameter(trial, img_size_param, pretrained_model_path, optuna_suggest,
                            batch_size_suggest_higher_bound=8, lr_suggest_bound=None, noise_steps_bound=None,
                            sampling_steps_bound=None, accumulate_grad_batches_bound=None,
                            max_noise_steps_sampling_bound=None, alt_params=None):
    if lr_suggest_bound is None:
        lr_suggest_bound = [1e-6, 1e-2]

    if noise_steps_bound is None:
        noise_steps_bound = [250, 1000]

    if sampling_steps_bound is None:
        sampling_steps_bound = [1, 25]

    if accumulate_grad_batches_bound is None:
        accumulate_grad_batches_bound = [1, 25]

    if alt_params is None:
        alt_params = {"batch_size": 4, "learning_rate": 1e-5, "optimizer_name": "AdamW",
                      "noise_steps": 1000, "sampling_steps": 2, "max_noise_steps_sampling": 400,
                      "accumulate_grad_batches": 1}

    if max_noise_steps_sampling_bound is None:
        max_noise_steps_sampling_bound = [400, alt_params["noise_steps"] - 1]

    accumulate_grad_batches = setuphelper_optuna_int(trial, optuna_suggest,
                                                     "accumulate_grad_batches", accumulate_grad_batches_bound,
                                                     alt_params)

    noise_steps = setuphelper_optuna_int(trial, optuna_suggest,
                                         "noise_steps", noise_steps_bound, alt_params)

    sampling_steps = setuphelper_optuna_int(trial, optuna_suggest,
                                            "sampling_steps", sampling_steps_bound, alt_params)

    max_noise_steps_sampling = setuphelper_optuna_int(trial, optuna_suggest, "max_noise_steps_sampling",
                                                      max_noise_steps_sampling_bound, alt_params)

    if optuna_suggest["batch_size_suggest"]:
        batch_size = trial.suggest_int("batch_size", 1, batch_size_suggest_higher_bound)
    else:
        batch_size = alt_params["batch_size"]

    if optuna_suggest["lr_suggest"]:
        learning_rate = trial.suggest_float("learning_rate", lr_suggest_bound[0],
                                            lr_suggest_bound[1], step=lr_suggest_bound[0] * 50)
        if learning_rate > lr_suggest_bound[0]:
            learning_rate -= lr_suggest_bound[0]
    else:
        learning_rate = alt_params["learning_rate"]

    if optuna_suggest["optimizer_suggest"]:
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
    else:
        optimizer_name = alt_params["optimizer_name"]

    # Create the PyTorch Lightning model with the hyperparameters.
    model_hparams = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer_name": optimizer_name,
        "img_size": img_size_param,
        "noise_steps": noise_steps,
        "sampling_steps": sampling_steps,
        "max_noise_steps_sampling": max_noise_steps_sampling,
        "pretrained_model_path": pretrained_model_path,
        "accumulate_grad_batches": accumulate_grad_batches,
    }

    if model_hparams["max_noise_steps_sampling"] >= model_hparams["noise_steps"]:
        value = model_hparams["max_noise_steps_sampling"]
        raise ValueError(f"max_noise_steps_sampling needs to be smaller than noise_steps {value}.")

    data_transforms = setup_datatransforms(model_hparams["img_size"])

    return model_hparams, data_transforms
