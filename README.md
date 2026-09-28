# Toward Channel-Robust and Receiver-Independent Radio Frequency Fingerprint Identification

## Overview

Radio frequency fingerprint identification (RFFI) identifies wireless devices from hardware-dependent signal characteristics. In practice, variations in propagation channels and receiver hardware can obscure these fingerprints, while collecting large labeled training datasets across devices is costly. To address these challenges, the paper proposes a three-stage approach for LoRa RFFI:

1. **Unsupervised pretraining:** learn an RF feature extractor using contrastive learning and augmented views of unlabeled signals.
2. **Training with contrastive loss:** train a classifier using signals from two receivers, pairing samples from the same transmitter across receivers and combining classification and contrastive losses.
3. **Inference:** identify a transmitter from its received signal.

The implementation includes signal augmentation with multipath fading and additive noise, spectrogram preprocessing, a residual CNN feature extractor, and a receiver-adversarial comparison model. The paper reports evaluations on three public LoRa datasets and one self-collected LoRa dataset.

## Repository contents

| File | Purpose |
| --- | --- |
| `main_pre.py` | Unsupervised contrastive pretraining. |
| `train_rffi_multirec.py` | Multi-receiver transmitter classification with contrastive pairs. |
| `train_rffi_oregonDataset.py` | Training on the public Oregon dataset. |
| `reverse_rec.py` | Receiver-adversarial comparison experiment. |
| `dlmodels.py` | Feature extractor and classification models. |
| `utils.py` | HDF5 loading, augmentation, pairing, and preprocessing. |
| `simclr.py` | Contrastive loss and training utilities. |
| `sr_pytorch.py` | Additional spectrogram utilities. |
| `test_sup_learn_set.py`, `test_plot_all.ipynb` | Evaluation and plotting experiments. |

## Requirement

- Python 3.10
- Pytorch

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{ma2025channelrobust,
  title   = {Toward Channel-Robust and Receiver-Independent Radio Frequency Fingerprint Identification},
  author  = {Ma, Jie and Zhang, Junqing and Shen, Guanxiong and Peng, Linning and Marshall, Alan},
  journal = {IEEE Transactions on Information Forensics and Security},
  volume  = {20},
  pages   = {12112--12125},
  year    = {2025},
  doi     = {10.1109/TIFS.2025.3630316}
}
```
