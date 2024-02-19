import copy

import optuna
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image


# Source: https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", batch_size=4,
                 sampling_steps=1, max_noise_steps_sampling=400, dtype_param=torch.float16, t_0_mode="linear"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.batch_size = batch_size
        self.sampling_steps = sampling_steps
        self.dtype_param = dtype_param
        self.t_0_mode = t_0_mode
        self.max_noise_steps_sampling = max_noise_steps_sampling

        self.T_0, self.T_0_next = self.prepare_t_0s()

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def prepare_t_0_linear(self):
        seq = np.linspace(0, 1, self.sampling_steps) * self.max_noise_steps_sampling
        return [int(s) for s in list(seq)]

    def prepare_t_0s(self):
        if self.t_0_mode != "linear":
            raise ValueError(f"Unknown t_0_mode {self.t_0_mode}.")
        T_0 = self.prepare_t_0_linear()
        T_0_next = [-1] + list(T_0[:-1])
        size = (self.batch_size,)
        return [torch.full(size=size, fill_value=value, device=self.device) for value in T_0], \
               [torch.full(size=size, fill_value=value, device=self.device) for value in T_0_next]

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def clip_diffusion_step(self, pred_image, t, pred_noise):
        # t needs to be prev or next
        if t.sum() == -t.shape[0]:
            # tensor: (1,)
            at_next = torch.ones_like(t)
            sqrt_alpha_hat = torch.sqrt(at_next)[:, None, None, None]
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - at_next)[:, None, None, None]
        else:
            test = self.alpha_hat[t]
            sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return (sqrt_alpha_hat * pred_image + sqrt_one_minus_alpha_hat * pred_noise).to(self.dtype_param)

    def predict_ground_truth(self, i_t_noise, t, noise):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return ((1 / sqrt_alpha_hat) * (i_t_noise - sqrt_one_minus_alpha_hat * noise)).to(self.dtype_param)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)


class DDPM_Model(pl.LightningModule):
    def __init__(self, conf, c_in=7, c_out=3, time_dim=256, *args, **kwargs):
        super().__init__()
        self.hparams.update(conf)
        self.diffusion = Diffusion(img_size=self.hparams.img_size)
        self.save_hyperparameters(ignore=["diffusion", "c_in"])
        self.sa1_image_size = int(self.hparams.img_size / 2 ** 3)
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)  # [4, 32, 256, 256] DoubleConv(c_in, output-inc)
        self.down1 = Down(64, 128, emb_dim=time_dim)  # [4, 64, 128, 128] Down(output-inc, output-down1)
        self.down2 = Down(128, 256, emb_dim=time_dim)  # [4, 128, 64, 64] Down(output-down1, output-down2)
        self.down3 = Down(256, 512, emb_dim=time_dim)  # [4, 256, 32, 32] Down(output-down2, output-down3)
        self.sa1 = SelfAttention(512,
                                 self.sa1_image_size)  # [4, 256, 32, 32] SelfAttention(output-down3, img_size/(2) ** 3)
        self.up1 = Up(768, 512, emb_dim=time_dim)  # [4, 128, 64, 64] (output-down3+output-down2, output-down3)
        self.up2 = Up(640, 128, emb_dim=time_dim)  # [4, 64, 128, 128] (output-down3+output-down1, output-down1)
        self.up3 = Up(192, 64, emb_dim=time_dim)  # [4, 32, 256, 256] (output-down1+output-inc, output-inc)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        t = t.unsqueeze(-1)  # (4,)(623,291,21,247) --> (4,1) [623],[291],[21],[247]
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1).to(self.device)

    def forward(self, x, t):
        t = self.pos_encoding(t, self.time_dim)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        att_block = self.sa1(x4)

        x = self.up1(att_block, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)

        return self.outc(x)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        if loss is not None:
            metrics = {"train_loss": loss}
            self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        if loss is not None:
            metrics = {"val_loss": loss}
            self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        if loss is not None:
            metrics = {"test_loss": loss}
            self.log_dict(metrics)
        return loss

    def _shared_step(self, batch, batch_idx):
        grayscale_reference_image, colored_image = batch
        t = self.diffusion.sample_timesteps(colored_image.shape[0])
        noised_img, noise = self.diffusion.noise_images(colored_image, t)
        tensor_concat = torch.cat((grayscale_reference_image, noised_img), dim=1)
        predicted_noise = self.forward(tensor_concat, t)
        return F.mse_loss(predicted_noise, noise)

    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer_name
        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "RMSprop": torch.optim.RMSprop,
            "SGD": torch.optim.SGD
        }
        if optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer name {optimizer_name}.")
        lr = self.hparams.learning_rate
        if optimizer_name in ["SGD", "RMSprop"]:
            return optimizers[optimizer_name](self.parameters(), lr=lr, momentum=0.9)
        else:
            return optimizers[optimizer_name](self.parameters(), lr=lr)


class DDPM_Model_finetune(pl.LightningModule):
    def __init__(self, conf, every_n_samples_loss=1,  *args, **kwargs):
        super().__init__()
        self.every_n_samples_loss = every_n_samples_loss
        self.hparams.update(conf)
        try:
            self.diffusion = Diffusion(img_size=self.hparams.img_size, noise_steps=self.hparams.noise_steps, sampling_steps=self.hparams.sampling_steps, batch_size=self.hparams.batch_size, max_noise_steps_sampling=self.hparams.max_noise_steps_sampling)
        except AttributeError:
            self.diffusion = Diffusion(img_size=self.hparams.img_size, noise_steps=self.hparams.noise_steps,
                                    sampling_steps=self.hparams.sampling_steps, batch_size=self.hparams.batch_size)

        self.T_0 = self.diffusion.T_0
        self.T_0_next = self.diffusion.T_0_next
        self.save_hyperparameters(ignore=["models", "forward_noise_model", "diffusion", "c_in"])
        self.model = DDPM_Model.load_from_checkpoint(self.hparams.pretrained_model_path)
        self.forward_noise_model = copy.deepcopy(self.model).eval()
        self.forward_noise_model.freeze()
        self.automatic_optimization = False

    def forward(self, x, t):
        return self.model.forward(x, t)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        if loss is not None:
            metrics = {"train_loss": loss}
            self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        if loss is not None:
            metrics = {"val_loss": loss}
            self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        if loss is not None:
            metrics = {"test_loss": loss}
            self.log_dict(metrics)
        return loss

    def _forward_noise(self, grayscale_reference_image):
        with torch.no_grad():
            I_t_noise = grayscale_reference_image[:, 1:4, :, :]
            I_t_noise = I_t_noise.to(torch.float16)
            for t, t_prev in zip((self.T_0_next[1:]), (self.T_0[1:])):
                tensor_concat = torch.cat((grayscale_reference_image, I_t_noise), dim=1)
                # Calculate the noise with the pretrained model
                pred_theta = self.forward_noise_model.forward(tensor_concat, t)
                # Predict the ground truth image from the noisy image and the noise
                I_gt_hat = self.diffusion.predict_ground_truth(I_t_noise, t, pred_theta)
                # Forward step: mix the noisy image with the predicted image and the noise
                I_t_noise = self.diffusion.clip_diffusion_step(I_gt_hat, t_prev, pred_theta)
                # forward_tensor_list.append(I_t_noise)
        return I_t_noise

    def _reverse_noise(self, grayscale_reference_image, I_gt_noise, colored_image):
        counter = 0
        # index 0   1   2   3   4!   5   6   7   8   9!
        # steps 999 888 776 666 555 444 333 222 111 0
        for t, t_next in zip(reversed(self.T_0), reversed(self.T_0_next)):
            if counter == self.every_n_samples_loss:
                I_gt_noise = I_gt_noise.detach()
                counter = 0
            tensor_concat = torch.cat((grayscale_reference_image, I_gt_noise), dim=1)
            # Calculate the noise with the pretrained model
            pred_theta = self.forward(tensor_concat, t)
            # Predict the ground truth image from the noisy image and the noise
            I_gt_hat = self.diffusion.predict_ground_truth(I_gt_noise, t, pred_theta)
            # Reverse step: mix the noisy image with the predicted image and the noise
            I_gt_noise = self.diffusion.clip_diffusion_step(I_gt_hat, t_next, pred_theta)
            if I_gt_hat.requires_grad and counter == self.every_n_samples_loss-1:
                opt = self.optimizers()
                opt.zero_grad()
                loss = F.mse_loss(colored_image, I_gt_noise)
                self.manual_backward(loss)
                opt.step()
            counter += 1
        # Update reconstruction loss: measure how well the fine-tuned model can restore the original image
        return loss if I_gt_hat.requires_grad else F.mse_loss(colored_image, I_gt_noise)

    def _shared_step(self, batch, batch_idx, opt=None):
        grayscale_reference_image, colored_image = batch
        grayscale_reference_image = grayscale_reference_image.to(torch.float16)

        I_gt_noise = self._forward_noise(grayscale_reference_image)
        I_gt_noise = I_gt_noise.to(torch.float16)
        return self._reverse_noise(grayscale_reference_image, I_gt_noise, colored_image)


    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer_name
        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "RMSprop": torch.optim.RMSprop,
            "SGD": torch.optim.SGD
        }
        if optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer name {optimizer_name}.")
        lr = self.hparams.learning_rate
        if optimizer_name in ["SGD", "RMSprop"]:
            return optimizers[optimizer_name](self.parameters(), lr=lr, momentum=0.9)
        else:
            return optimizers[optimizer_name](self.parameters(), lr=lr)

    def inverse_normalize(self, x):
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def plot_images_from_list(self, T_0, tensor_list):
        transform = T.ToPILImage()
        # create figure
        rows = 2
        columns = 6
        fig = plt.figure(figsize=(10, 7))
        for counter, (element, t) in enumerate(zip(tensor_list, T_0)):
            denoise_output = transform(
                self.inverse_normalize(element[0].detach().clone()))
            # Create a new subplot and add the image to it
            fig.add_subplot(rows, columns, counter + 1)
            plt.imshow(denoise_output)
            plt.axis('off')
            plt.title(f"{t[0]}")
        plt.show()
