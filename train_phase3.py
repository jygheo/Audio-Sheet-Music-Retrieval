import os
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torchvision.transforms as T
import torchaudio.transforms as T_audio
from tqdm import tqdm

from dataset import MSMDDataset, class_vocab, custom_collate_fn, get_deterministic_splits
from models import SpectrogramSwin, SheetMusicSwin, SymmetricCrossModalMoCo, VisionAudioMoCo
from utils import set_seed, load_checkpoint, evaluate_retrieval_phase2

def train_phase_3():
    set_seed()
    DATASET_ROOT = './msmd_dataset/msmd_aug_v1-1_no-audio'
    save_dir = './checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_pieces, val_pieces, _ = get_deterministic_splits(DATASET_ROOT)

    train_loader = DataLoader(MSMDDataset(DATASET_ROOT, train_pieces, class_vocab), batch_size=16, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)
    val_loader = DataLoader(MSMDDataset(DATASET_ROOT, val_pieces, class_vocab, mode='val'), batch_size=16, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)

    # Bootstrapping Phase 1 Audio and Phase 2 Vision weights
    audio_encoder = SpectrogramSwin(out_channels=512).to(device)
    dummy_moco = SymmetricCrossModalMoCo(SheetMusicSwin(), audio_encoder, K=1).to(device)
    dummy_moco, *_ = load_checkpoint(os.path.join(save_dir, "phase1_moco_best.pth"), dummy_moco, device=device)
    audio_encoder = dummy_moco.encoder_q_audio

    vision_student = SheetMusicSwin(out_channels=512).to(device)
    vision_student, *_ = load_checkpoint(os.path.join(save_dir, "phase2_vision_student_best.pth"), vision_student, device=device)

    moco_model = VisionAudioMoCo(vision_student, audio_encoder, K=16384).to(device)
    optimizer = optim.AdamW(moco_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, 1e-6, 1.0, 1000), CosineAnnealingLR(optimizer, 29)], [1])

    vision_augmenter = T.Compose([
        T.RandomAdjustSharpness(2, 0.5), T.ColorJitter(0.2, 0.2), T.RandomAffine(1, (0.02, 0.02))
    ])

    best_loss = float('inf')

    for epoch in range(30):
        moco_model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/30")
        
        freq_mask = T_audio.FrequencyMasking(15)
        time_mask = T_audio.TimeMasking(30)

        for batch_data in progress_bar:
            images = vision_augmenter(batch_data['images'].to(device))
            audio_inputs = batch_data['spectrograms'].to(device)
            audio_inputs = torch.stack([time_mask(freq_mask(x)) for x in audio_inputs.unbind(0)])

            optimizer.zero_grad()
            loss = moco_model(images, audio_inputs)
            loss.backward()
            clip_grad_norm_(moco_model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': moco_model.state_dict(), 'loss': best_loss}, os.path.join(save_dir, "phase3_moco_best.pth"))

        metrics = evaluate_retrieval_phase2(moco_model.encoder_q_vision, moco_model.encoder_q_audio, val_loader, device)
        print(metrics)

if __name__ == "__main__":
    train_phase_3()