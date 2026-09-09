# Dataset

DIF-SEG uses two optional datasets depending on the experiment.

## Segmentation

Place paired image/mask files here:

```text
dataset/
├── images/
│   ├── image_001.png
│   └── image_002.png
└── masks/
    ├── image_001.png
    └── image_002.png
```

Image and mask filenames must match. Masks should contain background and foreground regions.

Train the U-Net:

```bash
python -m segmentation.train
```

The best checkpoint is saved locally as `models/unet_best.pt`.

## Detection

For binary abnormality classification:

```text
dataset/
└── detection/
    ├── normal/
    │   ├── image_001.png
    │   └── image_002.png
    └── abnormal/
        ├── image_003.png
        └── image_004.png
```

Train the CNN baseline:

```bash
python -m detection.train
```

The best checkpoint is saved locally as `models/detection_best.pt`.

## Data safety

Use a properly licensed dataset. Do not commit private, identifiable, restricted, or otherwise sensitive medical data to a public repository.
