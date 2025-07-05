import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.t2m_dataset import Text2MotionDataset
from models.motion_clip import MotionTransformer
from data.dataset_motion_loader import get_dataset_motion_loader
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
import numpy as np
from os.path import join as pjoin
import argparse

import torch.nn.functional as F

# Define encode_motion function
def encode_motion(self, motion_data):
    """
    Encodes motion data into a 512-d embedding.

    Args:
        motion_data (torch.Tensor): Tensor of shape (batch_size, 196, 263)
        temporal_mask (torch.Tensor): Mask for temporal data.

    Returns:
        torch.Tensor: Motion embeddings of shape (batch_size, 512)
    """
    motion_features = self.motion(motion_data.float())  # Forward pass through MotionTransformer
    motion_features = motion_features / motion_features.norm(dim=-1, keepdim=True)  # Normalize features
    return motion_features

# Load dataset
def load_t2m_dataset(opt):
    """Loads the Text2Motion dataset with proper normalization."""
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt=opt, mean=mean, std=std, split_file=train_split_file)
    val_dataset = Text2MotionDataset(opt=opt, mean=mean, std=std, split_file=val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=8, shuffle=False, drop_last=True, pin_memory=True)

    return train_loader, val_loader

def validate_model(student_model, val_loader, teacher_model, epoch, device, unfreeze_epoch=35):
    student_model.eval()
    total_val_loss = 0.0
    total_dist_teacher = 0.0
    total_dist_student = 0.0
    total_dist_teacher_student = 0.0
    motion_alignment = 0.0

    with torch.no_grad():
        for conds, motion, m_lens in tqdm(val_loader, desc="Validation"):
            motion = motion.to(device).float()
            joint = recover_from_ric(motion, opt.joints_num).to(device)
            texts = clip.tokenize(conds).to(device)

            motion_features = student_model.encode_motion(joint)
            text_features_student = student_model.encode_text(texts).float()
            text_features_student = F.normalize(text_features_student, dim=-1)
            motion_features = F.normalize(motion_features, dim=-1)
            
            
            if epoch >= unfreeze_epoch:
                text_features_teacher = teacher_model.encode_text(texts).float().detach()
                text_features_teacher = F.normalize(text_features_teacher, dim=-1)
                distillation_loss = F.mse_loss(text_features_student, text_features_teacher)
                dist_teacher = F.mse_loss(motion_features, text_features_teacher).item()
            else:
                distillation_loss = 0.0
                dist_teacher = 0.0
                dist_teacher_student = 0.0


            logit_scale = student_model.logit_scale.exp()
            logits_per_motion = logit_scale * motion_features @ text_features_student.T
            logits_per_text = logits_per_motion.T
            labels = torch.arange(motion.shape[0], device=device)

            contrastive_loss = (F.cross_entropy(logits_per_motion, labels) +
                                F.cross_entropy(logits_per_text, labels)) / 2
            
            
            motion_alignment_loss = 1 - F.cosine_similarity(text_features_student, motion_features, dim=-1).mean()

            loss = contrastive_loss + lambda_distill * distillation_loss + motion_alignment_loss
            total_val_loss += loss.item()

            # Compute distances
            
            dist_student = F.mse_loss(motion_features, text_features_student).item()
            
            total_dist_teacher += dist_teacher
            total_dist_student += dist_student
            total_dist_teacher_student += distillation_loss
            motion_alignment += motion_alignment_loss
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_dist_teacher = total_dist_teacher / len(val_loader)
    avg_dist_student = total_dist_student / len(val_loader)
    avg_dist_teacher_student = total_dist_teacher_student / len(val_loader)
    avg_motion_alignment = motion_alignment / len(val_loader)

    print(f"Validation Loss: {avg_val_loss:.6f}")
    print(f"Distance from Teacher: {avg_dist_teacher:.6f}")
    print(f"Distance from Student: {avg_dist_student:.6f}")
    print(f"Distance from Teacher to Student: {avg_dist_teacher_student:.6f}")
    print(f"Alignment to Motion {avg_motion_alignment:.6f}")
    return avg_val_loss


if __name__ == '__main__':
    # Load CLIP model
    parser = argparse.ArgumentParser(description="Train motion CLIP model")
    parser.add_argument("--lda", type=float, default=0.6, help="Distillation loss weight")
    parser.add_argument("--dataset_name", type=str, default="t2m", help="Dataset name (e.g., 't2m', 'kit')")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for training")

    args = parser.parse_args()
    lambda_distill = args.lda
    dataset_name = args.dataset_name
    gpu_id = args.gpu_id

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Load training options
    dataset_opt_path = f'./checkpoints/{dataset_name}/Comp_v6_KLD005/opt.txt'
    opt = get_opt(dataset_opt_path, torch.device(device))
    opt.dataset_name = dataset_name

    opt.batch_size = 128 if opt.dataset_name == 't2m' else 32

    if opt.dataset_name == 't2m':
        opt.vq_name = "rvq_nq6_dc512_nc512_noshare_qdp0.2"
        opt.data_root = './dataset/HumanML3D'
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == 'kit': 
        opt.vq_name = "rvq_nq6_dc512_nc512_noshare_qdp0.2_k"
        opt.data_root = './dataset/KIT-ML'
        opt.text_dir = './dataset/KIT-ML/texts'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'

    else:
        print()
        raise KeyError('Dataset Does Not Exist')

    # Load pre-trained CLIP (Teacher & Student)
    teacher_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    student_model.float()

    # Add custom MotionTransformer
    student_model.motion = MotionTransformer(opt.dataset_name).to(device)
    student_model.encode_motion = encode_motion.__get__(student_model, type(student_model))

    for param in student_model.parameters():
        param.requires_grad = False

    # Unfreeze motion model parameters (they will be trained from the start)
    for param in student_model.motion.parameters():
        param.requires_grad = True

    # Optimizer setup with differential LR
    optimizer = torch.optim.AdamW([
        {'params': student_model.motion.parameters(), 'lr': 1e-4},
        {'params': student_model.text_projection, 'lr': 1e-5},
        {'params': student_model.token_embedding.parameters(), 'lr': 1e-6},
        {'params': student_model.positional_embedding, 'lr': 1e-6},
        {'params': student_model.transformer.parameters(), 'lr': 1e-6}
    ], weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(
    #     [{'params': student_model.motion.parameters(), 'lr': 5e-6, 'name': 'motion'}],
    #     weight_decay=0.01
    # )

    # Load dataset
    train_loader, val_loader = load_t2m_dataset(opt)

    num_epochs = 50
    progressive_unfreeze_start = 30  # After this epoch, start fine-tuning CLIP
    num_layers_to_unfreeze = 4  # How many layers to progressively unfreeze
    print("lambda_distill:", lambda_distill)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training Loop with Progressive Unfreezing
    progressive_unfreeze_start = 0
    num_layers_to_unfreeze = 4
    num_layers_total = len(list(student_model.transformer.resblocks))
    layers_per_step = num_layers_to_unfreeze // (num_epochs - progressive_unfreeze_start)
    unfreeze_epoch = 35

    # Training Loop with Validation
    for epoch in range(num_epochs):
        student_model.train()
        total_loss = 0.0
        total_dist_teacher = 0.0
        total_dist_student = 0.0
        total_dist_teacher_student = 0.0
        motion_alignment = 0.0

        if epoch == unfreeze_epoch:
            for param in student_model.parameters():
                param.requires_grad = True  # Unfreeze text encoder

            # Update optimizer to include text encoder parameters
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-6, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            student_model.train()


        if (epoch) % 5 == 0:
            validate_model(student_model, val_loader, teacher_model, epoch, device, unfreeze_epoch)

        for conds, motion, m_lens in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            motion = motion.to(device).float()
            joint = recover_from_ric(motion, opt.joints_num).to(device)
            texts = clip.tokenize(conds).to(device)

            optimizer.zero_grad()

            motion_features = student_model.encode_motion(joint)
            text_features_student = student_model.encode_text(texts).float()
            
            text_features_student = F.normalize(text_features_student, dim=-1)
            motion_features = F.normalize(motion_features, dim=-1)
            
            if epoch >= unfreeze_epoch:
                with torch.no_grad():
                    text_features_teacher = teacher_model.encode_text(texts).float().detach()
                text_features_teacher = F.normalize(text_features_teacher, dim=-1)
                distillation_loss = F.mse_loss(text_features_student, text_features_teacher)
                dist_teacher = F.mse_loss(motion_features, text_features_teacher).item()
            else:
                distillation_loss = 0.0
                dist_teacher = 0.0
                dist_teacher_student = 0.0
            

            logit_scale = student_model.logit_scale.exp()
            logits_per_motion = logit_scale * motion_features @ text_features_student.T
            logits_per_text = logits_per_motion.T
            labels = torch.arange(motion.shape[0], device=device)

            contrastive_loss = (F.cross_entropy(logits_per_motion, labels) +
                                F.cross_entropy(logits_per_text, labels)) / 2
            
            
            motion_alignment_loss = 1 - F.cosine_similarity(text_features_student, motion_features, dim=-1).mean()

            loss = contrastive_loss + lambda_distill * distillation_loss + motion_alignment_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Compute distances
            
            dist_student = F.mse_loss(motion_features, text_features_student).item()
            
            total_dist_teacher += dist_teacher
            total_dist_student += dist_student
            total_dist_teacher_student += distillation_loss
            motion_alignment += motion_alignment_loss

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.6f}, \
              Distance from Teacher: {total_dist_teacher / len(train_loader):.6f}, \
              Distance from Student: {total_dist_student / len(train_loader):.6f}, \
              Distance from Teacher to Student: {total_dist_teacher_student / len(train_loader):.6f} \
              Alignment to Motion: {motion_alignment / len(train_loader):.6f}")

    # Save trained model
    # torch.save(student_model.state_dict(), "moCLIP_l10.pth")
    lambda_str = str(int(lambda_distill * 10)).zfill(2)  # Converts 1.0 -> "10", 0.75 -> "07", etc.
    base_path = f"./checkpoints/{opt.dataset_name}/moCLIP/"
    os.makedirs(base_path, exist_ok=True)
    save_path = os.path.join(base_path, f"moCLIP_l{lambda_str}_{dataset_name}.pth")
    torch.save(student_model.state_dict(), save_path)

    print("Training complete.")
