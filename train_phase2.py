import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torchvision.transforms as T
from tqdm import tqdm

from dataset import MSMDDataset, class_vocab, custom_collate_fn, get_deterministic_splits
from models import SheetMusicTeacherGAT, SpectrogramSwin, SheetMusicSwin, SymmetricCrossModalMoCo
from utils import set_seed, load_checkpoint, evaluate_retrieval_phase2

def train_phase_2():
    set_seed()
    DATASET_ROOT = './msmd_dataset/msmd_aug_v1-1_no-audio'
    save_dir = './checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_pieces, val_pieces, _ = get_deterministic_splits(DATASET_ROOT)

    train_loader = DataLoader(MSMDDataset(DATASET_ROOT, train_pieces, class_vocab), batch_size=16, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)
    val_loader = DataLoader(MSMDDataset(DATASET_ROOT, val_pieces, class_vocab, mode='val'), batch_size=16, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)

    # Load frozen Phase 1 models
    graph_encoder = SheetMusicTeacherGAT(num_classes=72).to(device)
    audio_encoder = SpectrogramSwin(out_channels=512).to(device)
    moco_model_old = SymmetricCrossModalMoCo(graph_encoder, audio_encoder, K=16384).to(device)
    moco_model_old, *_ = load_checkpoint(os.path.join(save_dir, "phase1_moco_best.pth"), moco_model_old, device=device)
    
    graph_teacher = moco_model_old.encoder_q_graph.eval()
    audio_teacher = moco_model_old.encoder_q_audio.eval()
    for p in graph_teacher.parameters(): p.requires_grad = False
    for p in audio_teacher.parameters(): p.requires_grad = False

    vision_student = SheetMusicSwin(out_channels=512).to(device)
    optimizer = optim.AdamW(vision_student.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, 1e-6, 1.0, 1000), CosineAnnealingLR(optimizer, 29)], [1])

    vision_augmenter = T.Compose([
        T.RandomAdjustSharpness(2, 0.5), T.ColorJitter(0.2, 0.2), T.RandomAffine(1, (0.02, 0.02))
    ])

    best_loss = float('inf')

    for epoch in range(30):
        vision_student.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/30")

        for batch_data in progress_bar:
            images = vision_augmenter(batch_data['images'].to(device))
            graph_inputs = {
                'x_cont': batch_data['graph_x_cont'].to(device), 'x_class': batch_data['graph_x_class'].to(device),
                'x_pitch': batch_data['graph_x_pitch'].to(device), 'edge_index': batch_data['graph_edge_index'].to(device),
                'batch': batch_data['graph_batch_index'].to(device)
            }

            optimizer.zero_grad()
            with torch.no_grad(): target_g_embed = graph_teacher(**graph_inputs)
            
            v_embed = vision_student(images)
            loss = 1.0 - F.cosine_similarity(v_embed, target_g_embed, dim=-1).mean()
            loss.backward()
            clip_grad_norm_(vision_student.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': vision_student.state_dict(), 'loss': best_loss}, os.path.join(save_dir, "phase2_vision_student_best.pth"))

        metrics = evaluate_retrieval_phase2(vision_student, audio_teacher, val_loader, device)
        print(metrics)

if __name__ == "__main__":
    train_phase_2()