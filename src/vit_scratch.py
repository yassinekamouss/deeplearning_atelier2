"""
Part 2: Vision Transformer (ViT) Built From Scratch in PyTorch
Implementation following the step-by-step tutorial by Brian Pulfer:
https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

Lab 2: Deep Learning — FST Tanger
Student: Yassine Kamouss (LSI)
"""

import math
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torchvision import datasets, transforms
from data_utils import get_mnist_loaders, get_mnist_transforms


def patchify(images, n_patches):
    """
    Splits images (B, C, H, W) into n_patches x n_patches non-overlapping square patches
    and flattens each patch into a 1D vector.
    Returns: (B, n_patches**2, patch_dim)
    """
    n, c, h, w = images.shape
    assert h == w, "Patchify method implemented for square images."
    patch_size = h // n_patches
    
    # Vectorized patch extraction
    # images: (B, C, H, W) -> unfold -> (B, C, n_p, n_p, p_s, p_s)
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(n, n_patches * n_patches, -1)
    return patches


def get_positional_embeddings(sequence_length, d):
    """
    Computes standard sinusoidal positional embeddings (Vaswani et al.).
    Returns tensor of shape (sequence_length, d).
    """
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            if j % 2 == 0:
                result[i, j] = math.sin(i / (10000 ** (j / d)))
            else:
                result[i, j] = math.cos(i / (10000 ** ((j - 1) / d)))
    return result


class MyMSA(nn.Module):
    """
    Multi-Head Self-Attention (MSA) module built from scratch.
    """
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        assert d % n_heads == 0, f"Dimension {d} must be divisible by {n_heads} heads"
        self.d_head = d // n_heads
        
        self.q_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)])
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(d, d)
        
    def forward(self, sequences, return_attention=False):
        # sequences shape: (N, seq_len, d)
        result = []
        attentions = []
        
        for head in range(self.n_heads):
            q_mapping = self.q_mappings[head]
            k_mapping = self.k_mappings[head]
            v_mapping = self.v_mappings[head]
            
            # Slice per-head slice
            seq_head = sequences[:, :, head * self.d_head: (head + 1) * self.d_head]
            q = q_mapping(seq_head)
            k = k_mapping(seq_head)
            v = v_mapping(seq_head)
            
            # Scaled Dot-Product Attention: A = Softmax(Q * K^T / sqrt(d_head))
            attention = self.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_head))
            head_out = torch.bmm(attention, v)
            result.append(head_out)
            if return_attention:
                attentions.append(attention)
                
        # Concatenate across heads
        concat_out = torch.cat(result, dim=-1)
        out = self.out_proj(concat_out)
        
        if return_attention:
            return out, torch.stack(attentions, dim=1)  # (N, n_heads, seq_len, seq_len)
        return out


class MyViTBlock(nn.Module):
    """
    Standard Transformer Encoder Block:
    Pre-LN -> MSA -> Residual Connection -> Pre-LN -> MLP -> Residual Connection
    """
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super().__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.msa = MyMSA(hidden_d, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )
        
    def forward(self, x, return_attention=False):
        norm_x = self.norm1(x)
        if return_attention:
            msa_out, attn = self.msa(norm_x, return_attention=True)
            x = x + msa_out
            x = x + self.mlp(self.norm2(x))
            return x, attn
        else:
            x = x + self.msa(norm_x)
            x = x + self.mlp(self.norm2(x))
            return x


class MyViT(nn.Module):
    """
    Complete Vision Transformer (ViT) implementation from scratch.
    """
    def __init__(self, chw=(1, 28, 28), n_patches=7, n_blocks=2, hidden_d=64, n_heads=4, out_d=10):
        super().__init__()
        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        assert chw[1] % n_patches == 0 and chw[2] % n_patches == 0, "Image size must be divisible by n_patches"
        self.patch_size = chw[1] // n_patches
        self.input_d = int(chw[0] * (self.patch_size ** 2))
        
        # 1. Linear projection of flattened patches
        self.linear_mapper = nn.Linear(self.input_d, hidden_d)
        
        # 2. Learnable [CLS] classification token
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_d) * 0.02)
        
        # 3. Positional embeddings
        self.register_buffer(
            'positional_embeddings',
            get_positional_embeddings(n_patches ** 2 + 1, hidden_d),
            persistent=False
        )
        
        # 4. Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)
        ])
        
        # 5. Final Classification Head
        self.norm = nn.LayerNorm(hidden_d)
        self.mlp_head = nn.Linear(hidden_d, out_d)
        
    def forward(self, images, return_attention=False):
        n, c, h, w = images.shape
        # Extract patches
        patches = patchify(images, self.n_patches)  # (N, n_patches^2, input_d)
        tokens = self.linear_mapper(patches)       # (N, n_patches^2, hidden_d)
        
        # Prepend [CLS] token
        cls_tokens = self.class_token.expand(n, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (N, n_patches^2 + 1, hidden_d)
        
        # Add positional embedding
        tokens = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Pass through Transformer encoder blocks
        last_attention = None
        for block in self.blocks:
            if return_attention:
                tokens, last_attention = block(tokens, return_attention=True)
            else:
                tokens = block(tokens)
                
        # Extract [CLS] token representation for classification
        cls_feature = self.norm(tokens[:, 0])
        logits = self.mlp_head(cls_feature)
        
        if return_attention:
            return logits, last_attention
        return logits


def train_vit_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate_vit(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss, acc, f1, np.array(all_labels), np.array(all_preds)


def visualize_vit_attention(model, dataset, device, save_path: str = './figures/vit_attention_maps.png'):
    """
    Visualizes the self-attention maps of the [CLS] token attending to image patches.
    """
    model.eval()
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
    
    with torch.no_grad():
        for idx in range(6):
            img_tensor, label = dataset[idx * 20]
            img_input = img_tensor.unsqueeze(0).to(device)
            logits, attn = model(img_input, return_attention=True)
            pred = logits.argmax(dim=1).item()
            
            # attn shape: (1, n_heads, seq_len, seq_len)
            # Average attention over heads for [CLS] token (row 0) attending to patch tokens (cols 1..)
            cls_attn = attn[0].mean(dim=0)[0, 1:].cpu().numpy()
            n_patches = model.n_patches
            attn_grid = cls_attn.reshape(n_patches, n_patches)
            
            # Plot original digit
            axes[0, idx].imshow(img_tensor.squeeze().numpy(), cmap='gray')
            axes[0, idx].set_title(f"True: {label} | Pred: {pred}", fontsize=10, fontweight='bold')
            axes[0, idx].axis('off')
            
            # Plot attention heatmap
            sns.heatmap(attn_grid, cmap='viridis', ax=axes[1, idx], cbar=False)
            axes[1, idx].set_title("CLS Attention Map", fontsize=10)
            axes[1, idx].axis('off')
            
    plt.suptitle("Vision Transformer: Attention Focus on MNIST Patches", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_vit_training(epochs: int = 3, lr: float = 0.003, max_train_samples: int = 15000,
                     max_test_samples: int = 3000, device=None, save_dir: str = './figures'):
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training Vision Transformer (ViT) on Device: {device} | Epochs: {epochs}")
    
    train_loader, test_loader = get_mnist_loaders(batch_size=128, augment=False,
                                                  max_train_samples=max_train_samples,
                                                  max_test_samples=max_test_samples)
    test_ds = datasets.MNIST(root='./data', train=False, transform=get_mnist_transforms())
    
    # 7x7 patches of size 4x4, hidden_d=64, 4 heads, 2 transformer blocks
    model = MyViT(chw=(1, 28, 28), n_patches=7, n_blocks=2, hidden_d=64, n_heads=4, out_d=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    start_time = time.time()
    for ep in range(1, epochs + 1):
        t_loss, t_acc = train_vit_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_f1, _, _ = evaluate_vit(model, test_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['test_loss'].append(v_loss)
        history['test_acc'].append(v_acc)
        print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {t_loss:.4f} - Acc: {t_acc*100:.2f}% | Test Loss: {v_loss:.4f} - Acc: {v_acc*100:.2f}%")
        
    total_time = time.time() - start_time
    print(f"[+] ViT Training Completed in {total_time:.2f} seconds.")
    
    final_loss, final_acc, final_f1, y_true, y_pred = evaluate_vit(model, test_loader, criterion, device)
    
    # Visualizations
    visualize_vit_attention(model, test_ds, device, save_path=os.path.join(save_dir, 'vit_attention_maps.png'))
    
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], 'g-o', label='Train Loss', lw=2)
    plt.plot(range(1, epochs + 1), history['test_loss'], 'r--s', label='Test Loss', lw=2)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Cross-Entropy Loss', fontsize=11)
    plt.title('ViT Scratch: Loss vs Epochs', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history['train_acc']], 'g-o', label='Train Acc', lw=2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history['test_acc']], 'r--s', label='Test Acc', lw=2)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('ViT Scratch: Accuracy vs Epochs', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_training_curves.png'), dpi=300)
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'ViT Scratch Confusion Matrix (Acc: {final_acc*100:.2f}%)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Digit', fontsize=11)
    plt.ylabel('Actual Digit', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_confusion_matrix.png'), dpi=300)
    plt.close()
    
    return {
        'model': model,
        'accuracy': final_acc,
        'f1_score': final_f1,
        'test_loss': final_loss,
        'training_time': total_time,
        'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


if __name__ == '__main__':
    run_vit_training(epochs=5)
