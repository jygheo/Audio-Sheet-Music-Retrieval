import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

class SheetMusicTeacherGAT(nn.Module):
    def __init__(self, num_classes, num_pitches=129, embed_dim=64, hidden_channels=128, out_channels=512, heads=4):
        super(SheetMusicTeacherGAT, self).__init__()
        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embed_dim)
        self.pitch_embedding = nn.Embedding(num_embeddings=num_pitches, embedding_dim=embed_dim)
        
        in_channels = 6 + (embed_dim * 2)
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, 256, heads=1, concat=False, dropout=0.1)
        self.norm2 = nn.LayerNorm(256)

        self.projection_head = nn.Sequential(
            nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, out_channels)
        )

    def forward(self, x_cont, x_class, x_pitch, edge_index, batch):
        class_emb = self.class_embedding(x_class)
        pitch_emb = self.pitch_embedding(x_pitch)
        x = torch.cat([x_cont, class_emb, pitch_emb], dim=1)

        x = F.elu(self.norm1(self.gat1(x, edge_index)))
        x = F.elu(self.norm2(self.gat2(x, edge_index)))

        x_pooled = global_mean_pool(x, batch)
        out = F.normalize(self.projection_head(x_pooled), p=2, dim=1)
        return out

class SheetMusicSwin(nn.Module):
    def __init__(self, out_channels=512):
        super(SheetMusicSwin, self).__init__()
        self.backbone = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, out_channels)
        )

    def forward(self, x):
        features = self.backbone(x)
        return F.normalize(self.projection_head(features), p=2, dim=1)

class SpectrogramSwin(nn.Module):
    def __init__(self, out_channels=512):
        super(SpectrogramSwin, self).__init__()
        self.backbone = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        
        original_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(1, original_conv.out_channels, original_conv.kernel_size, original_conv.stride, original_conv.padding)
        new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        if original_conv.bias is not None:
            new_conv.bias.data = original_conv.bias.data
        self.backbone.features[0][0] = new_conv

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, out_channels)
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        x = F.pad(x, (0, (32 - W % 32) % 32, 0, (32 - H % 32) % 32))
        return F.normalize(self.projection_head(self.backbone(x)), p=2, dim=1)

class SymmetricCrossModalMoCo(nn.Module):
    def __init__(self, graph_encoder, audio_encoder, dim=512, K=65536, m=0.999, T=0.07):
        super(SymmetricCrossModalMoCo, self).__init__()
        self.K, self.m, self.T = K, m, T
        self.encoder_q_graph, self.encoder_q_audio = graph_encoder, audio_encoder
        self.encoder_k_graph, self.encoder_k_audio = copy.deepcopy(graph_encoder), copy.deepcopy(audio_encoder)

        for param_k in self.encoder_k_graph.parameters(): param_k.requires_grad = False
        for param_k in self.encoder_k_audio.parameters(): param_k.requires_grad = False

        self.register_buffer("queue_graph", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_audio", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoders(self):
        for p_q, p_k in zip(self.encoder_q_graph.parameters(), self.encoder_k_graph.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)
        for p_q, p_k in zip(self.encoder_q_audio.parameters(), self.encoder_k_audio.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_graph, keys_audio):
        batch_size = keys_graph.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            overflow = (ptr + batch_size) - self.K
            fit = batch_size - overflow
            self.queue_graph[:, ptr:self.K] = keys_graph.T[:, :fit]
            self.queue_audio[:, ptr:self.K] = keys_audio.T[:, :fit]
            self.queue_graph[:, 0:overflow] = keys_graph.T[:, fit:]
            self.queue_audio[:, 0:overflow] = keys_audio.T[:, fit:]
        else:
            self.queue_graph[:, ptr:ptr + batch_size] = keys_graph.T
            self.queue_audio[:, ptr:ptr + batch_size] = keys_audio.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, graph_inputs, audio_inputs):
        self._momentum_update_key_encoders()
        q_graph = self.encoder_q_graph(**graph_inputs)
        q_audio = self.encoder_q_audio(audio_inputs)

        with torch.no_grad():
            k_graph = self.encoder_k_graph(**graph_inputs)
            k_audio = self.encoder_k_audio(audio_inputs)

        l_pos_G2A = torch.einsum('nc,nc->n', [q_graph, k_audio]).unsqueeze(-1)
        l_neg_G2A = torch.einsum('nc,ck->nk', [q_graph, self.queue_audio.clone().detach()])
        logits_G2A = torch.cat([l_pos_G2A, l_neg_G2A], dim=1) / self.T

        l_pos_A2G = torch.einsum('nc,nc->n', [q_audio, k_graph]).unsqueeze(-1)
        l_neg_A2G = torch.einsum('nc,ck->nk', [q_audio, self.queue_graph.clone().detach()])
        logits_A2G = torch.cat([l_pos_A2G, l_neg_A2G], dim=1) / self.T

        labels = torch.zeros(logits_G2A.shape[0], dtype=torch.long).to(q_graph.device)
        loss = F.cross_entropy(logits_G2A, labels) + F.cross_entropy(logits_A2G, labels)

        self._dequeue_and_enqueue(k_graph, k_audio)
        return loss

class VisionAudioMoCo(nn.Module):
    def __init__(self, vision_encoder, audio_encoder, dim=512, K=16384, m=0.999, T=0.07):
        super(VisionAudioMoCo, self).__init__()
        self.K, self.m, self.T = K, m, T
        self.encoder_q_vision, self.encoder_q_audio = vision_encoder, audio_encoder
        self.encoder_k_vision, self.encoder_k_audio = copy.deepcopy(vision_encoder), copy.deepcopy(audio_encoder)

        for param_k in self.encoder_k_vision.parameters(): param_k.requires_grad = False
        for param_k in self.encoder_k_audio.parameters(): param_k.requires_grad = False

        self.register_buffer("queue_vision", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_audio", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoders(self):
        for p_q, p_k in zip(self.encoder_q_vision.parameters(), self.encoder_k_vision.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)
        for p_q, p_k in zip(self.encoder_q_audio.parameters(), self.encoder_k_audio.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_vision, keys_audio):
        batch_size = keys_vision.shape[0]
        ptr = int(self.queue_ptr)
        self.queue_vision[:, ptr:ptr + batch_size] = keys_vision.T
        self.queue_audio[:, ptr:ptr + batch_size] = keys_audio.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, images, audio_inputs):
        self._momentum_update_key_encoders()
        q_vision = self.encoder_q_vision(images)
        q_audio = self.encoder_q_audio(audio_inputs)

        with torch.no_grad():
            k_vision = self.encoder_k_vision(images)
            k_audio = self.encoder_k_audio(audio_inputs)

        l_pos_V2A = torch.einsum('nc,nc->n', [q_vision, k_audio]).unsqueeze(-1)
        l_neg_V2A = torch.einsum('nc,ck->nk', [q_vision, self.queue_audio.clone().detach()])
        logits_V2A = torch.cat([l_pos_V2A, l_neg_V2A], dim=1) / self.T

        l_pos_A2V = torch.einsum('nc,nc->n', [q_audio, k_vision]).unsqueeze(-1)
        l_neg_A2V = torch.einsum('nc,ck->nk', [q_audio, self.queue_vision.clone().detach()])
        logits_A2V = torch.cat([l_pos_A2V, l_neg_A2V], dim=1) / self.T

        labels = torch.zeros(logits_V2A.shape[0], dtype=torch.long).to(q_vision.device)
        loss = F.cross_entropy(logits_V2A, labels) + F.cross_entropy(logits_A2V, labels)
        self._dequeue_and_enqueue(k_vision, k_audio)
        return loss