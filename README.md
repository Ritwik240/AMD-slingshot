
 **Soyabean Rust Segmentation Using U‑Net**

This repository contains a deep learning pipeline for **soyabean rust disease segmentation** — a semantic segmentation model that identifies and highlights rust spots on soybean leaf images.

The model uses a **U‑Net architecture** trained on an annotated dataset to predict pixel‑level masks that delineate where rust occurs on the leaf. This is useful for precision agriculture, early disease detection, and automated crop health monitoring.


## Project Structure

soyabean‑rust‑segmentation/
│
├── dataset/
│ ├── images/ # Original leaf images
│ └── masks/ # Corresponding binary rust masks
│
├── train.py # Training script
├── infer.py # Batch inference script
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── results/ # Inference output (mask & overlay)

##  Download Trained Model

The pretrained U‑Net weights (`.pth` file) are not stored in this repo due to GitHub file size limits.

👉 Download the model here:

📦 **[best_soy_rust_unet.pth](https://drive.google.com/file/d/1XVjUZuSv3MC4OK3dqp7ibK1VwZNiKZce/view?usp=drive_link)**

After downloading, place it in the **root of this repository** (same level as `train.py` and `infer.py`).

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt

Required packages include:

PyTorch
torchvision
OpenCV
NumPy
Albumentations
scikit‑learn
tqdm

----
Training

To train the model on your dataset:

python dataset.py

----
Inference (Batch)

Place the images you want to test into a folder, for example:
test_images/
   leaf1.jpg
   leaf2.jpg
   ...

Then run:
python infer.py

This will generate for each image:

[name]_original.png → Original image

[name]_mask.png → Predicted binary mask

[name]_overlay.png → Rust overlay highlight

Results are saved in the results/ folder.


How It Works

Dataset Loading – Uses Albumentations for augmentation & preprocessing.
U‑Net Model – Encoder‑decoder architecture with skip connections.
Training – Combines BCEWithLogits + Dice loss.
Inference – Sigmoid activation + thresholding to produce binary masks.
Overlay – Highlights rust regions red on the original leaf image.
