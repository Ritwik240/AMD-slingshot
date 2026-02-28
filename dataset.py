import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Dataset
# -------------------------------------------------
class SoyRustDataset(Dataset):
    def __init__(self, images, masks, img_size=256, train=True):
        self.images = images
        self.masks = masks
        self.size = img_size

        if train:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Resize(self.size, self.size),
                A.Normalize(mean=(0.485,0.456,0.406),
                            std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Resize(self.size, self.size),
                A.Normalize(mean=(0.485,0.456,0.406),
                            std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found: {self.masks[idx]}")
        mask = (mask > 0).astype(np.float32)

        augmented = self.aug(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"].unsqueeze(0)

        return img, mask

# -------------------------------------------------
# U-Net
# -------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.u1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.c1 = DoubleConv(1024, 512)

        self.u2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c2 = DoubleConv(512, 256)

        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c3 = DoubleConv(256, 128)

        self.u4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        b = self.bottleneck(self.pool(d4))

        x = self.u1(b)
        x = self.c1(torch.cat([x, d4], dim=1))

        x = self.u2(x)
        x = self.c2(torch.cat([x, d3], dim=1))

        x = self.u3(x)
        x = self.c3(torch.cat([x, d2], dim=1))

        x = self.u4(x)
        x = self.c4(torch.cat([x, d1], dim=1))

        return self.out(x)

# -------------------------------------------------
# Dice Loss
# -------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.*intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

# -------------------------------------------------
# Training
# -------------------------------------------------
def train():
    imgs = sorted(glob("dataset/images/*"))
    masks = sorted(glob("dataset/masks/*"))

    train_i, val_i, train_m, val_m = train_test_split(
        imgs, masks, test_size=0.2, random_state=42
    )

    train_ds = SoyRustDataset(train_i, train_m, train=True)
    val_ds = SoyRustDataset(val_i, val_m, train=False)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    model = UNet().to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = 999

    for epoch in range(120):
        model.train()
        train_loss = 0

        loop = tqdm(train_loader)
        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = 0.5*bce(preds, masks) + dice(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        # -------- Validation --------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                loss = bce(preds, masks) + dice(preds, masks)
                val_loss += loss.item()

        print(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_soy_rust_unet.pth")
            print("✅ Best model saved")

if __name__ == "__main__":
    train()