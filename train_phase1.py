import os
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torchaudio.transforms as T_audio
from tqdm import tqdm

from dataset import MSMDDataset, class_vocab, custom_collate_fn, get_deterministic_splits
from models import SheetMusicTeacherGAT, SpectrogramSwin, SymmetricCrossModalMoCo
from utils import set_seed, seed_worker, load_checkpoint, evaluate_retrieval

def train_phase_1():
    set_seed()
    g = torch.Generator()
    g.manual_seed(42)

    DATASET_ROOT = './msmd_dataset/msmd_aug_v1-1_no-audio'
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_pieces, val_pieces, _ = get_deterministic_splits(DATASET_ROOT)

    train_loader = DataLoader(
        MSMDDataset(DATASET_ROOT, train_pieces, class_vocab),
        batch_size=16, shuffle=True, collate_fn=custom_collate_fn,
        num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        MSMDDataset(DATASET_ROOT, val_pieces, class_vocab, mode='val'),
        batch_size=16, shuffle=False, collate_fn=custom_collate_fn,
        num_workers=2, pin_memory=True, worker_init_fn=seed_worker
    )

    graph_encoder = SheetMusicTeacherGAT(num_classes=72).to(device)
    audio_encoder = SpectrogramSwin(out_channels=512).to(device)
    moco_model = SymmetricCrossModalMoCo(graph_encoder, audio_encoder, K=16384).to(device)

    optimizer = optim.AdamW(moco_model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, 1e-6, 1.0, 1000), CosineAnnealingLR(optimizer, 19)], [1])
    
    best_loss = float('inf')

    for epoch in range(20):
        moco_model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20")

        freq_mask = T_audio.FrequencyMasking(15)
        time_mask = T_audio.TimeMasking(30)

        for batch_data in progress_bar:
            audio_inputs = batch_data['spectrograms'].to(device)
            audio_inputs = torch.stack([time_mask(freq_mask(x)) for x in audio_inputs.unbind(0)])

            graph_inputs = {
                'x_cont': batch_data['graph_x_cont'].to(device), 'x_class': batch_data['graph_x_class'].to(device),
                'x_pitch': batch_data['graph_x_pitch'].to(device), 'edge_index': batch_data['graph_edge_index'].to(device),
                'batch': batch_data['graph_batch_index'].to(device)
            }
            
            jitter = (torch.rand_like(graph_inputs['x_cont'][:, :2]) - 0.5) * 0.02
            graph_inputs['x_cont'][:, :2] = torch.clamp(graph_inputs['x_cont'][:, :2] + jitter, 0.0, 1.0)

            optimizer.zero_grad()
            loss = moco_model(graph_inputs, audio_inputs)
            loss.backward()
            clip_grad_norm_(moco_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': moco_model.state_dict(), 'loss': best_loss}, os.path.join(save_dir, "phase1_moco_best.pth"))

        metrics = evaluate_retrieval(moco_model, val_loader, device)
        print(metrics)

if __name__ == "__main__":
    train_phase_1()