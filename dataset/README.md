# Real dataset: ISIC 2016 Part 3B

DIF-SEG is configured to use a real, publicly documented medical-image benchmark rather than synthetic demo data.

The project uses the **ISIC 2016 Part 3B** training set from the International Skin Imaging Collaboration (ISIC). It contains 900 dermoscopic lesion images, paired lesion-segmentation masks, and a 900-row ground-truth CSV labeling each case as `benign` or `malignant`. The ISIC documentation states that the masks were created by expert clinicians and the malignancy labels came from expert consensus and pathology report information.

Official dataset page: https://challenge.isic-archive.com/data/

## Download and prepare

From the repository root:

```bash
python -m scripts.download_isic2016
python -m scripts.prepare_isic2016
```

The downloader retrieves the official ISIC archive files into `dataset/raw/isic2016/`. The preparation script then creates the folders expected by the training code.

## Resulting structure

```text
dataset/
├── images/
│   ├── ISIC_*.jpg
│   └── ...
├── masks/
│   ├── ISIC_*.png
│   └── ...
├── detection/
│   ├── normal/
│   │   └── ISIC_*.jpg
│   └── abnormal/
│       └── ISIC_*.jpg
└── raw/
    └── isic2016/
        ├── ISBI2016_ISIC_Part3B_Training_Data.zip
        └── ISBI2016_ISIC_Part3B_Training_GroundTruth.csv
```

`dataset/raw/`, `dataset/images/`, `dataset/masks/`, and `dataset/detection/` are ignored by Git, so the medical dataset is not committed to the public repository.

## Train

Segmentation:

```bash
python -m segmentation.train
```

Detection:

```bash
python -m detection.train
```

Checkpoints are written locally to `models/` and are also ignored by Git.

## Evaluation

Use a held-out validation/test split and report metrics from actual predictions. The segmentation module currently provides Dice and IoU. The ISIC 2016 challenge documentation also describes sensitivity, specificity, accuracy, Jaccard, and Dice for lesion segmentation, while the classification task reports sensitivity, specificity, accuracy, and average precision.

## Licensing and citation

Check the current ISIC terms before redistribution or publication. The ISIC 2016 dataset is listed as CC-0 on the official ISIC dataset page.

Do not add private patient data, restricted datasets, or identifiable medical records to this repository.
