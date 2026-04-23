# Contrastive Audio-Sheet Music Retrieval

This project maps piano sheet music images and audio spectrograms into a shared embedding space for cross-modal retrieval.   
Training: [colab](https://drive.google.com/file/d/1E6JfUSSxqE5v19OKsW20HSeVMKslJB8k/view?usp=sharing)

## Architecture and Training 
Training pipeline relies heavily on the structural annotations provided by the MSMD (Multimodal Sheet Music Dataset) [1].

### Data  & Graph Construction
To speed up cross-modal alignment, we use the structural information from MSMD XML files to create directed graphs representing the musical score. 

* **Node Features:** Each note is represented as a node. Nodes encode continuous features (normalized bounding box `[top, left, width, height]`, normalized onset time, and duration) as well as categorical embeddings (MIDI pitch).
* **Edge Construction:** The adjacency matrix is built using standard XML outlinks, and additional bidirectional edges for simultaneous notes (chords) and forward-directed edges connecting directly adjacent notes over time.
* **Image and Audio Preprocessing:** Full sheet music pages are cropped into individual staff lines. Audio spectrograms are sliced into discrete time chunks, nomalized and padded to match batch dimensions. During training, images are augmented with affine transformations and color/sharpness jittering, while spectrograms use SpecAugment (time and frequency masking).

### Training
1.  **Phase 1 (Graph-Audio pretraining):** A Graph Attention Network (GAT) and a Spectrogram Swin Transformer are trained with a contrastive objective to align graph structure with audio representations
2.  **Phase 2 (Graph-Vision distillation):** An Image Swin Transformer is trained to match the frozen GAT embeddings using a cosine similarity loss, transferring structural information from the graph into the vision model
3.  **Phase 3 (Audio-Vision training):** The audio and vision Swin Transformers are jointly fine-tuned to better align their representations
4.  
### Contrastive Setup (MoCo)
Cross-modal alignment in Phases 1 and 3 uses Momentum Contrast to maintain a large number of negative samples.
* **Queue Size (K):** 16384
* **Momentum (m):** 0.999
* **Temperature (τ):** 0.07

### Evaluation Metrics
Cross-modal retrieval performance (Audio-to-Score and Score-to-Audio) is measured using the following ranking metrics:
* **R@1:** 
* **R@5:** 
* **R@25:**
* **MRR (Mean Reciprocal Rank):** 
* **MR (Median Rank):** 

---
**References**
[1] Matthias Dorfer, Jan Hajič jr., Andreas Arzt, Harald Frostel, Gerhard Widmer. *Learning Audio-Sheet Music Correspondences for Cross-Modal Retrieval and Piece Identification.* Transactions of the International Society for Music Information Retrieval, issue 1, 2018.


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
## Dataset
The data pipeline relies on the MSMD dataset (Augmented v1.1). Ensure it's unzipped and structured properly in your directory.

```bash
# Example Download
wget "https://zenodo.org/record/2597505/files/msmd_aug_v1-1_no-audio.zip"
unzip msmd_aug_v1-1_no-audio.zip -d ./msmd_dataset
```

## Running the Training Pipeline
Ensure your directory contains a ./checkpoints folder to hold intermediate .pth files. To train, run the phases sequentially:

Preprocess the Images
```bash
python preprocess_images.py
```

Phase 1
```bash
python train_phase1.py
```
Phase 2
```bash
python train_phase2.py
```
Phase 3
```bash
python train_phase3.py
```
Get Inference Checkpoints
```bash
python export_weights.py
```
