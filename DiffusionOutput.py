import pathlib

import PIL.Image
import torch

from DiffusionModel import DDPM_Model, Diffusion, DDPM_Model_finetune
from setuphandler import setup_datatransforms
from DataModule import SketchAndColorDataModule
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pytorch_lightning import seed_everything
import os
from PIL import Image

animesketchcolorizationpair_train = pathlib.Path("./dataset/animeface/train")
animesketchcolorizationpair_test = pathlib.Path("./dataset/animeface/val")

model_path = "./models/DDPM-Model-Finetuning-128-Face/trial-6.ckpt"
amount = 100

folderpath = "./modelinput/"

lineart_path = "./modelinput/Lineart.jpg"
original_path = "./modelinput/Color.jpg"
reference_path = "./modelinput/Reference.jpg"

single_image_mode = False
folder_image_mode = False
pred_save = False

title = model_path.replace('./models/', '').replace('/trial-', '-trial').replace('.ckpt', '')
model_hparams = {
    "img_size": 128,
    "batch_size": 10,
    "noise_steps": 1000,
    "sampling_steps": 25,
    "max_noise_steps_sampling": 999
}

seed = 22

ddpm_model = False

flip = True


def inverse_normalize(x):
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x


def display_all_images(grayscale_reference_image, pred_image, original_image, batch_size, title="color prediction"):
    transform = T.ToPILImage()
    reference_image = grayscale_reference_image[:, 1:4, :, :]
    grayscale_image = grayscale_reference_image[:, :1, :, :]
    for i in range(batch_size):
        # create figure
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle(title)
        # setting values to rows and column variables
        rows = 1
        columns = 4
        # lineart
        color = transform(inverse_normalize(grayscale_image[i].detach().clone()))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(color, cmap='gray')
        plt.axis("off")
        plt.title("lineart")

        # reference
        reference = transform(inverse_normalize(reference_image[i].detach().clone()))
        fig.add_subplot(rows, columns, 2)
        plt.imshow(reference)
        plt.axis("off")
        plt.title("reference")

        # prediction
        pred = transform(inverse_normalize(pred_image[i].detach().clone()))
        fig.add_subplot(rows, columns, 3)
        plt.imshow(pred)
        plt.axis("off")
        plt.title("prediction")

        # original
        original = transform(inverse_normalize(original_image[i].detach().clone()))
        fig.add_subplot(rows, columns, 4)
        plt.imshow(original)
        plt.axis("off")
        plt.title("original")

        # Displaying the plot
        plt.show()


def plot_images(colored_image, I_gt_hat, batch_size, title, left_column_title, right_column_title):
    transform = T.ToPILImage()
    # create figure
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(title)
    # setting values to rows and column variables
    rows = batch_size
    columns = 2
    for i in range(rows):
        color = inverse_normalize(colored_image[i].detach().clone())
        color = transform(color)
        fig.add_subplot(rows, columns, 1 + 2 * i)
        plt.imshow(color)
        plt.axis("off")
        plt.title(left_column_title)

        reference = inverse_normalize(I_gt_hat[i].detach().clone())
        reference = transform(reference)
        fig.add_subplot(rows, columns, 2 + 2 * i)
        plt.imshow(reference)
        plt.axis("off")
        plt.title(right_column_title)

    # Displaying the plot
    plt.show()


def plot_images_from_list_denoising_process(tensor_list, T_0):
    transform = T.ToPILImage()
    # create figure
    rows = 5
    columns = 5
    fig = plt.figure(figsize=(10, 7))
    reverse_T = reversed(T_0)
    for counter, (element, t) in enumerate(zip(tensor_list, reverse_T)):
        denoise_output = transform(inverse_normalize(element[0].detach().clone()))
        # Create a new subplot and add the image to it
        fig.add_subplot(rows, columns, counter + 1)
        plt.imshow(denoise_output)
        plt.axis('off')
        plt.title(f"{t[0]}")
    plt.show()


def _sample_loop(T_0, T_0_next, denoising_images_output, model, diffusion, grayscale_reference_image, I_gt_noise, device="cuda"):
    for t, t_next in zip(reversed(T_0), reversed(T_0_next)):
        # train a copy of the model and combine the weights with the original model?
        tensor_concat = torch.cat((grayscale_reference_image.to(device), I_gt_noise), dim=1)
        # Calculate the noise with the pretrained model
        pred_theta = model.forward(tensor_concat, t)
        # Predict the ground truth image from the noisy image and the noise
        I_gt_hat = diffusion.predict_ground_truth(I_gt_noise, t, pred_theta)
        # Reverse step: mix the noisy image with the predicted image and the noise
        I_gt_noise = diffusion.clip_diffusion_step(I_gt_hat, t_next, pred_theta)
        denoising_images_output.append(I_gt_noise)
    return I_gt_noise, denoising_images_output


def sample(model, diffusion, grayscale_reference_image, output_de_noising_process=False, device="cuda"):
    denoising_images_output = []
    T_0_next = diffusion.T_0_next
    T_0 = diffusion.T_0
    reference_image = grayscale_reference_image[:, 1:4, :, :]
    with torch.no_grad():
        I_t_noise, _ = diffusion.noise_images(reference_image.to(device), T_0[-1])
        denoising_images_output.append(I_t_noise)
        if output_de_noising_process:
            plot_images(reference_image, I_t_noise, model_hparams["batch_size"], "noising_prep", "before", "after")
        I_gt_noise = I_t_noise
        I_gt_noise, denoising_images_output = _sample_loop(T_0, T_0_next, denoising_images_output, model, diffusion,
                                                           grayscale_reference_image, I_gt_noise)
        if output_de_noising_process:
            plot_images_from_list_denoising_process(denoising_images_output, T_0)
        return I_gt_noise


def prepare_batch(lineart_path_param, reference_path_param, original_path_param):
    lineart_img = data_transforms["grayscale"](Image.open(lineart_path_param).convert('L'))
    ref_img = data_transforms["reference"](Image.open(reference_path_param).convert('RGB'))
    original_img = data_transforms["colored"](Image.open(original_path_param).convert('RGB'))
    grayscale_reference_image = torch.cat((lineart_img, ref_img), dim=0)
    grayscale_reference_image = grayscale_reference_image[None, :, :, :]
    original_img = original_img[None, :, :, :]
    return grayscale_reference_image, original_img


if __name__ == "__main__":
    seed_everything(seed, True)
    data_transforms = setup_datatransforms(model_hparams["img_size"])
    if single_image_mode:
        Images = [prepare_batch(lineart_path, reference_path, original_path)]
    elif folder_image_mode:
        Images = []
        amount = 100
        num_images = 14
        for i in range(num_images):
            lineart_path = f"{folderpath}/Line-{i}.jpg"
            reference_path = f"{folderpath}/Ref-{i}.jpg"
            original_path = f"{folderpath}/Original-{i}.jpg"
            Images.append(prepare_batch(lineart_path, reference_path, original_path))


    else:
        dm = SketchAndColorDataModule(imagefolder_train=animesketchcolorizationpair_train,
                                      imagefolder_test=animesketchcolorizationpair_test, extension="png",
                                      batch_size=model_hparams["batch_size"],
                                      num_workers=8, transform=data_transforms)
        dm.setup("test")
        Images = dm.test_dataloader()  # dataloader
    if ddpm_model:
        model = DDPM_Model.load_from_checkpoint(model_path).to("cuda")
    else:
        model = DDPM_Model_finetune.load_from_checkpoint(model_path).to("cuda")  # pretrained noise prediction model
    model.eval()
    diffusion = Diffusion(noise_steps=model_hparams["noise_steps"], sampling_steps=model_hparams["sampling_steps"],
                          batch_size=model_hparams["batch_size"], t_0_mode="linear",
                          max_noise_steps_sampling=model_hparams["max_noise_steps_sampling"])

    torch.set_float32_matmul_precision("high")
    T_0 = diffusion.T_0

    with torch.no_grad():
        for counter, I_gt in enumerate(Images):  # 3 loop through each element image of dataloader
            grayscale_reference_image, original_image = I_gt
            reference_image = grayscale_reference_image[:, 1:4, :, :]
            if flip:
                reference_image = torch.flip(reference_image, dims=[0])
                grayscale_image = grayscale_reference_image[:, :1, :, :]
                grayscale_reference_image = torch.cat((grayscale_image, reference_image), dim=1)
            pred_image = sample(model, diffusion, grayscale_reference_image)
            display_all_images(grayscale_reference_image, pred_image, original_image, model_hparams["batch_size"],
                               title=title)
            if pred_save:
                transform = T.ToPILImage()
                pred = transform(inverse_normalize(pred_image[0].detach().clone()))
                pred.save(f"./modelinput/comparison/Own/Pred-{counter}.jpg")
            if counter == amount - 1:
                break
