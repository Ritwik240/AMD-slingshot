import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# Load model
# -------------------------------------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load("best_soy_rust_unet.pth", map_location=DEVICE))
model.eval()

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# -------------------------------------------------
# Paths
# -------------------------------------------------
input_folder = "test_images"   # folder with images for inference
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

image_paths = sorted(glob(os.path.join(input_folder, "*.*")))

threshold = 0.2

# -------------------------------------------------
# Inference loop
# -------------------------------------------------
for img_path in image_paths:
    image = cv2.imread(img_path)
    orig = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aug = transform(image=image_rgb)
    input_tensor = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred)

    mask = pred.squeeze(0).squeeze(0).cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255

    # Resize mask to original image size
    mask_resized = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

    # Create overlay
    overlay = orig.copy()
    overlay[mask_resized == 255] = [0, 0, 255]  # red overlay

    # Save outputs
    base_name = os.path.basename(img_path).split(".")[0]
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_original.png"), orig)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), mask_resized)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_overlay.png"), overlay)


print("✅ Inference completed. All results saved in 'results' folder.")