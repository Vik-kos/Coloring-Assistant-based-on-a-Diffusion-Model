from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from torch.utils.data import Dataset
import os
import pathlib
import torch
from torch_tps import ThinPlateSpline
from PIL import Image
import torchvision.transforms as T
# Importieren Sie die benötigten Bibliotheken
import numpy as np


def random_ctrl_points():
    """Generate random ctrl points
    # Source https://github.com/raphaelreme/torch-tps/blob/main/example/image_warping.py
    (In proportion of the desired shapes)
    """
    input_ctrl = torch.rand(10, 2)
    output_ctrl = input_ctrl + torch.randn(10, 2) * 0.05

    corners = torch.tensor(  # Add corners ctrl points
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    return torch.cat((input_ctrl, corners)), torch.cat((output_ctrl, corners))


def tps_transform(image):
    """Warp an image"""
    # Source: https://github.com/raphaelreme/torch-tps/blob/main/example/image_warping.py
    # Load the image
    width, height = image.size

    size = torch.tensor((height, width))

    # Build control points
    input_ctrl, output_ctrl = random_ctrl_points()
    input_ctrl *= size
    output_ctrl *= size

    # Fit the thin plate spline from output to input
    tps = ThinPlateSpline(0.5)
    tps.fit(output_ctrl, input_ctrl)

    # Create the 2d meshgrid of indices for output image
    i = torch.arange(height, dtype=torch.float32)
    j = torch.arange(width, dtype=torch.float32)

    ii, jj = torch.meshgrid(i, j, indexing="ij")
    output_indices = torch.cat((ii[..., None], jj[..., None]), dim=-1)  # Shape (H, W, 2)

    # Transform it into the input indices
    input_indices = tps.transform(output_indices.reshape(-1, 2)).reshape(height, width, 2)

    # Interpolate the resulting image
    grid = 2 * input_indices / size - 1  # Into [-1, 1]
    grid = torch.flip(grid, (-1,))  # Grid sample works with x,y coordinates, not i, j
    torch_image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)[None, ...]

    transformed_image_tensor = torch.nn.functional.grid_sample(torch_image, grid[None, ...], align_corners=False)[0]

    return T.ToPILImage()(transformed_image_tensor.to(torch.uint8))


class DatasetWithSketchIncluded(Dataset):
    def __init__(self, imagefolder: pathlib.Path, extension: str = "png", transform=None):
        self._imagefolder = imagefolder
        self.extension = extension
        self.transform = transform
        # Only calculate once how many files are in this folder
        self._length = sum(1 for _ in os.listdir(self._imagefolder))

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # images always follow [0, n-1], so you access them directly
        img = Image.open(self._imagefolder / f"{str(index)}.{self.extension}").convert('RGB')
        width, _ = img.size
        width_cutoff = width // 2
        img_array = np.array(img)
        colored_image = Image.fromarray(img_array[:, :width_cutoff])
        grayscale_image = Image.fromarray(img_array[:, width_cutoff:]).convert('L')
        reference_image = tps_transform(Image.fromarray(img_array[:, :width_cutoff]))
        if self.transform:
            grayscale_image = self.transform["grayscale"](grayscale_image)
            colored_image = self.transform["colored"](colored_image)
            reference_image = self.transform["reference"](reference_image)
        return torch.cat((grayscale_image, reference_image), dim=0), colored_image


class SketchAndColorDataModule(pl.LightningDataModule):

    def __init__(self, imagefolder_train: pathlib.Path, imagefolder_test: pathlib.Path, extension: str = "jpg",
                 batch_size: int = 32,
                 num_workers: int = 8, transform=None):
        super().__init__()
        self.imagefolder_train = imagefolder_train
        self.imagefolder_test = imagefolder_test
        self.extension = extension
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage: str):

        if stage == "fit" or stage is None:
            train_set_full = DatasetWithSketchIncluded(
                imagefolder=self.imagefolder_train, extension=self.extension, transform=self.transform)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = DatasetWithSketchIncluded(
                imagefolder=self.imagefolder_test, extension=self.extension, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=True, prefetch_factor=4, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=True, prefetch_factor=4, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=True, prefetch_factor=4, drop_last=True)
