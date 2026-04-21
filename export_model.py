import os
import torch
from models import SheetMusicSwin, SpectrogramSwin, VisionAudioMoCo
from utils import load_checkpoint

def export_final_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = './checkpoints'

    vision_student = SheetMusicSwin(out_channels=512).to(device)
    audio_encoder = SpectrogramSwin(out_channels=512).to(device)

    moco_model = VisionAudioMoCo(vision_encoder=vision_student, audio_encoder=audio_encoder, dim=512, K=16384).to(device)
    
    resume_checkpoint = os.path.join(save_dir, "phase3_moco_best.pth")
    moco_model, *_ = load_checkpoint(resume_checkpoint, moco_model, device=device)
    
    vision_student = moco_model.encoder_q_vision
    audio_encoder = moco_model.encoder_q_audio

    torch.save({
        'vision_state_dict': vision_student.state_dict(),
        'audio_state_dict': audio_encoder.state_dict()
    }, os.path.join(save_dir, "vision_audio_model.pth"))
    print("Final inference weights isolated and saved to vision_audio_model.pth")

if __name__ == "__main__":
    export_final_model()