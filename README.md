# RS-SCD: Semantic Change Detection in Remote Sensing Images

Code for semantic change detection (SCD) in bi-temporal remote sensing imagery, including SRNet and DFINet.

SCNet will be released soon. Stay tuned!  

SJH-SCD will be released soon. Stay tuned!  

## SRNet

SRNet is a Swin-based semantic change detection network that adapts a pretrained Swin backbone with Mona Adapter (MA, parameter-efficient fine-tuning) and refines the results with a progressive decoder.

- `models/Swin/SRNet.py` — SRNet (Swin-Base backbone).
- `models/Swin/SRNet_tiny.py` — SRNet-tiny (Swin-Tiny backbone).

### Citation

If you find SRNet useful in your research, please consider citing:

> Z. Jiang et al., "SRNet: Semantic Anchoring and Fine-Grained Refinement Network for Semantic Change Detection in Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 4413617-4413617, 2026, Art no. 4413617, doi: [10.1109/TGRS.2026.3715494](https://doi.org/10.1109/TGRS.2026.3715494).

```bibtex
@article{jiang2026srnet,
  title={SRNet: Semantic Anchoring and Fine-Grained Refinement Network for Semantic Change Detection in Remote Sensing Images},
  author={Jiang, Zhenghao and others},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={64},
  pages={4413617},
  year={2026},
  doi={10.1109/TGRS.2026.3715494}
}
```

### Pretrained weights (Baidu Pan)

Download the following files and place them at the given paths inside this repository:

| File | Place under | Baidu Pan link | Extract code |
|---|---|---|---|
| `SRNet_SECOND_23e_mIoU74.04_Sek25.15_Fscd63.93.pth` (trained SRNet) | `checkpoints/` | https://pan.baidu.com/s/14DYOpj_GzhDk6wFyJKrm1A | `2aej` |
| `SRNet_tiny_SECOND_28e_mIoU73.30_Sek23.84_Fscd62.79.pth` (trained SRNet-tiny) | `checkpoints/` | https://pan.baidu.com/s/1P4kLnmudtUK9bf5U-ZrVlg | `ceda` |
| `swin_base_patch4_window7_224_22k.pth` (ImageNet-22k pretrained backbone) | `models/Swin/weight/` | https://pan.baidu.com/s/1Z9fq_raz9yBYdKzixqrFSQ | `32e7` |
| `swin_tiny_patch4_window7_224_22k.pth` (ImageNet-22k pretrained backbone) | `models/Swin/weight/` | https://pan.baidu.com/s/14d1ZCwck0fDO1W3bGzogQg | `pux5` |

The Swin backbone weights are ImageNet-22k pretrained checkpoints used to initialize the frozen backbone (only the Mona Adapter and heads are trainable). The trained checkpoints are the released SRNet / SRNet-tiny models evaluated on SECOND.

### Repository layout (SRNet)

```
RS-SCD/
├── test_score.py              # evaluate SRNet on the SECOND test set
├── train_SRNet.py             # train SRNet-tiny from scratch
├── checkpoints/               # put the downloaded trained checkpoints here
├── models/
│   └── Swin/
│       ├── SRNet.py           # SRNet (Swin-Base + MA)
│       ├── SRNet_tiny.py      # SRNet-tiny (Swin-Tiny + MA)
│       └── weight/            # put the downloaded Swin pretrained backbones here
├── datasets/
│   └── RS_ST.py               # SECOND dataset loader (7 classes)
└── utils/                     # losses, evaluation metrics, etc.
```

### Train / Test

Environment: Python >= 3.8, PyTorch >= 1.9, plus `torchvision`, `timm`, `thop`, `tensorboardX`, `scikit-image`, `numpy`.

```bash
# Train SRNet-tiny on SECOND
python train_SRNet.py

# Evaluate SRNet (default model & checkpoint)
python test_score.py

# Explicit test directory / checkpoint
python test_score.py --test_dir "E:\SCD\SECOND_OG\test" \
    --chkpt_path "checkpoints/SRNet_SECOND_23e_mIoU74.04_Sek25.15_Fscd63.93.pth"
```

To evaluate SRNet-tiny, switch the model import in `test_score.py` to `from models.Swin.SRNet_tiny import SRNet as Net` and point `--chkpt_path` to `checkpoints/SRNet_tiny_SECOND_28e_mIoU73.30_Sek23.84_Fscd62.79.pth`.

Metrics reported: `Fscd`, `mIoU`, `SeK`, and pixel `Accuracy` on the test set.

## DFINet

The weights of DFINet on SECOND can be downloaded from Baidu Netdisk: Link: https://pan.baidu.com/s/1WlCOBc3L4_35j6Uolk5e5A  Extraction code: thif

## Datasets

Hi-UCD-mini can be downloaded from [Google Drive](https://drive.google.com/file/d/1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u/view). The above is the official link. We repartitioned the data following the method provided in Zuo, Xibing, et al. (2025) "Multitask Siamese Network Guided by Enhanced Change Information for Semantic Change Detection in Bi-temporal Remote Sensing Images." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 18:61–77. https://doi.org/10.1109/JSTARS.2024.3487137.
As described in the SCNet paper, the relevant link is: File shared via Baidu Netdisk — Hi-UCD-mini.zip
Link: https://pan.baidu.com/s/1jRPlJl6JklJgmjiHGMJ7TQ?pwd=gypg
Extraction code: gypg  

SECOND can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1fzAn4Bez_S6KX83iYABjAlASCzzhRJPQ).  

## Acknowledgement

Finally, we would like to express our sincere gratitude to the authors of the BiSRNet paper and for the reference code repository they provided.
