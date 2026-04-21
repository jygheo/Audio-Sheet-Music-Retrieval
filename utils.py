import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    print(f"Loading checkpoint from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    train_loss = checkpoint.get('loss', None)
    val_metrics = checkpoint.get('metrics', None)

    print(f"Successfully restored checkpoint from Epoch {epoch + 1}.")
    return model, optimizer, epoch, train_loss, val_metrics

@torch.no_grad()
def evaluate_retrieval(moco_model, val_loader, device='cuda'):
    moco_model.eval()
    all_graph_embeddings, all_audio_embeddings, all_line_ids = [], [], []

    for batch_data in tqdm(val_loader, desc="Validation Forward Pass"):
        audio_inputs = batch_data['spectrograms'].to(device)
        graph_inputs = {
            'x_cont': batch_data['graph_x_cont'].to(device),
            'x_class': batch_data['graph_x_class'].to(device),
            'x_pitch': batch_data['graph_x_pitch'].to(device),
            'edge_index': batch_data['graph_edge_index'].to(device),
            'batch': batch_data['graph_batch_index'].to(device)
        }

        q_graph = moco_model.encoder_q_graph(**graph_inputs)
        q_audio = moco_model.encoder_q_audio(audio_inputs)

        all_graph_embeddings.append(q_graph.cpu())
        all_audio_embeddings.append(q_audio.cpu())
        all_line_ids.extend(batch_data['line_id'])

    return _calc_metrics(all_graph_embeddings, all_audio_embeddings, all_line_ids, 'A2S', 'S2A')

@torch.no_grad()
def evaluate_retrieval_phase2(vision_student, audio_teacher, val_loader, device='cuda'):
    vision_student.eval()
    audio_teacher.eval()
    all_vision_embeddings, all_audio_embeddings, all_line_ids = [], [], []

    for batch_data in tqdm(val_loader, desc="Validation Forward Pass"):
        images = batch_data['images'].to(device)
        audio_inputs = batch_data['spectrograms'].to(device)

        q_audio = audio_teacher(audio_inputs)
        q_vision = vision_student(images)

        all_vision_embeddings.append(q_vision.cpu())
        all_audio_embeddings.append(q_audio.cpu())
        all_line_ids.extend(batch_data['line_id'])

    return _calc_metrics(all_vision_embeddings, all_audio_embeddings, all_line_ids, 'A2V', 'V2A')

def _calc_metrics(embeds_1, embeds_2, line_ids, name1, name2):
    e1 = torch.cat(embeds_1, dim=0)
    e2 = torch.cat(embeds_2, dim=0)
    sim_matrix = torch.matmul(e2, e1.T)

    labels_tensor = torch.tensor(LabelEncoder().fit_transform(line_ids))
    ground_truth_mask = (labels_tensor.unsqueeze(1) == labels_tensor.unsqueeze(0))

    def calculate_ranks(matrix):
        sorted_indices = torch.argsort(matrix, dim=1, descending=True)
        sorted_mask = torch.gather(ground_truth_mask, 1, sorted_indices)
        ranks = sorted_mask.float().argmax(dim=1) + 1
        return (ranks <= 1).float().mean().item(), (ranks <= 5).float().mean().item(), (1.0 / ranks).float().mean().item(), ranks.median().item()

    metrics = {}
    r1, r5, mrr, mr = calculate_ranks(sim_matrix)
    metrics[name1] = {'R@1': r1, 'R@5': r5, 'MRR': mrr, 'MR': mr}
    r1, r5, mrr, mr = calculate_ranks(sim_matrix.T)
    metrics[name2] = {'R@1': r1, 'R@5': r5, 'MRR': mrr, 'MR': mr}
    
    return metrics