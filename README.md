# Multimodal Sheet Music Contrastive Learning

This project establishes a joint embedding space between raw sheet music images, audio spectrograms, and XML-derived graph networks. It employs a three-phase training architecture leveraging Swin Transformers and Graph Attention Networks (GAT) alongside Contrastive Learning (MoCo).

## Project Structure

* `dataset.py`: Dataloaders for multimodal extraction from XML markup, Audio Spectrograms, and Images.
* `models.py`: Network architecture definitions (`SheetMusicTeacherGAT`, `SpectrogramSwin`, `SheetMusicSwin`, `SymmetricCrossModalMoCo`).
* `utils.py`: Metric evaluation, rank calculation, seed management, and checkpointing.
* `train_phase1.py`: Phase 1: Train Graph (GAT) & Audio (Swin) mapping via MoCo.
* `train_phase2.py`: Phase 2: Distill structural knowledge from Graph into Vision (Swin).
* `train_phase3.py`: Phase 3: Final MoCo finetuning on Vision & Audio directly.
* `export_weights.py`: Extract models for inference without the MoCo wrappers.

## Prerequisites

Python 3.8+ required.
Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install muscima tqdm scikit-learn
```
## Dataset Acquisition
The data pipeline relies on the MSMD dataset (Augmented v1.1). Ensure it's unzipped and structured properly in your directory.

```bash
# Example Download
wget "[https://zenodo.org/record/2597505/files/msmd_aug_v1-1_no-audio.zip](https://zenodo.org/record/2597505/files/msmd_aug_v1-1_no-audio.zip)"
unzip msmd_aug_v1-1_no-audio.zip -d ./msmd_dataset
```

## Running the Training Pipeline
Ensure your directory contains a ./checkpoints folder to hold intermediate .pth files. To train the network end-to-end, run the phases sequentially:

Phase 1: GNN-Audio Contrastive Training
```bash
python train_phase1.py
```
Phase 2: Graph-to-Vision Teacher/Student Distillation
```bash
python train_phase2.py
```
Phase 3: Joint Vision-Audio Fine-Tuning
```bash
python train_phase3.py
```
Isolate Inference Checkpoints
```bash
python export_weights.py
```
