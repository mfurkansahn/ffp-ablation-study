# FFP Baseline (PSNR-only)

This repository provides a **clean baseline implementation** of
Future Frame Prediction (FFP) for video anomaly detection.

## Purpose
This codebase is used as the **reference (A0) baseline** in thesis experiments.
It is intentionally kept minimal and unchanged to ensure fair comparison
with extended models.

## Baseline Characteristics
- Anomaly score: **PSNR only**
- No motion fusion
- No Bezier trajectory modeling
- No temporal regularization
- Architecture: Generator + Discriminator + FlowNetSD

## Environment
- Python: 3.8
- PyTorch: 1.13.1
- CUDA: 11.7
- Conda environment: `ffp_env`

### Dependencies
A minimal `requirements.txt` file is provided to document the main software
dependencies used in the experiments.

## Notes
- **Datasets are not included** (e.g., UCSD Ped2 must be provided separately).
- **Pretrained FlowNet weights are not included**.
- Dataset and model paths are defined in `config.py`.

## Reproducibility
All extended models are compared against this baseline under the **same
software environment and training settings**.
