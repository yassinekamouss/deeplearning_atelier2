"""
Part 1: Transfer Learning & Fine-Tuning with Pretrained AlexNet and VGG16
Lab 2: Deep Learning — FST Tanger
Student: Yassine Kamouss (LSI)
"""

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_mnist_loaders


def get_finetuned_alexnet(num_classes: int = 10, freeze_backbone: bool = True):
    """
    Load pretrained AlexNet and adapt classifier head for MNIST digits (10 classes).
    """
    model = models.alexnet(weights='DEFAULT')
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
            
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def get_finetuned_vgg16(num_classes: int = 10, freeze_backbone: bool = True):
    """
    Load pretrained VGG16 and adapt classifier head for MNIST digits (10 classes).
    """
    model = models.vgg16(weights='DEFAULT')
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
            
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def train_transfer_epoch(model, loader, criterion, optimizer, device):
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


def evaluate_transfer(model, loader, criterion, device):
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


def run_transfer_experiments(epochs: int = 2, lr: float = 0.0005, batch_size: int = 128,
                             max_train_samples: int = 10000, max_test_samples: int = 2000,
                             device=None, save_dir: str = './figures'):
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Resize to 64x64 and replicate to 3 channels for ImageNet architectures
    print("[*] Preparing 3-channel 64x64 MNIST loaders for Transfer Learning...")
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, augment=False,
                                                  three_channels=True, resize=64,
                                                  max_train_samples=max_train_samples,
                                                  max_test_samples=max_test_samples)
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    # 1. AlexNet
    print(f"\n[*] Fine-tuning AlexNet on {device}...")
    alexnet = get_finetuned_alexnet(num_classes=10, freeze_backbone=True).to(device)
    opt_alexnet = optim.Adam(filter(lambda p: p.requires_grad, alexnet.parameters()), lr=lr)
    
    start_time = time.time()
    alex_history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    for ep in range(1, epochs + 1):
        t_loss, t_acc = train_transfer_epoch(alexnet, train_loader, criterion, opt_alexnet, device)
        v_loss, v_acc, v_f1, _, _ = evaluate_transfer(alexnet, test_loader, criterion, device)
        alex_history['train_loss'].append(t_loss)
        alex_history['train_acc'].append(t_acc)
        alex_history['test_loss'].append(v_loss)
        alex_history['test_acc'].append(v_acc)
        print(f"  AlexNet Epoch {ep}/{epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc*100:.2f}% | Test Loss: {v_loss:.4f} Acc: {v_acc*100:.2f}%")
        
    alex_time = time.time() - start_time
    alex_loss, alex_acc, alex_f1, y_true, alex_preds = evaluate_transfer(alexnet, test_loader, criterion, device)
    
    results['AlexNet'] = {
        'model': alexnet,
        'accuracy': alex_acc,
        'f1_score': alex_f1,
        'test_loss': alex_loss,
        'training_time': alex_time,
        'parameters': sum(p.numel() for p in alexnet.parameters() if p.requires_grad),
        'total_params': sum(p.numel() for p in alexnet.parameters())
    }
    
    # 2. VGG16
    print(f"\n[*] Fine-tuning VGG16 on {device}...")
    vgg16 = get_finetuned_vgg16(num_classes=10, freeze_backbone=True).to(device)
    opt_vgg = optim.Adam(filter(lambda p: p.requires_grad, vgg16.parameters()), lr=lr)
    
    start_time = time.time()
    vgg_history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    for ep in range(1, epochs + 1):
        t_loss, t_acc = train_transfer_epoch(vgg16, train_loader, criterion, opt_vgg, device)
        v_loss, v_acc, v_f1, _, _ = evaluate_transfer(vgg16, test_loader, criterion, device)
        vgg_history['train_loss'].append(t_loss)
        vgg_history['train_acc'].append(t_acc)
        vgg_history['test_loss'].append(v_loss)
        vgg_history['test_acc'].append(v_acc)
        print(f"  VGG16 Epoch {ep}/{epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc*100:.2f}% | Test Loss: {v_loss:.4f} Acc: {v_acc*100:.2f}%")
        
    vgg_time = time.time() - start_time
    vgg_loss, vgg_acc, vgg_f1, _, vgg_preds = evaluate_transfer(vgg16, test_loader, criterion, device)
    
    results['VGG16'] = {
        'model': vgg16,
        'accuracy': vgg_acc,
        'f1_score': vgg_f1,
        'test_loss': vgg_loss,
        'training_time': vgg_time,
        'parameters': sum(p.numel() for p in vgg16.parameters() if p.requires_grad),
        'total_params': sum(p.numel() for p in vgg16.parameters())
    }
    
    # Visualization comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), alex_history['test_loss'], 'b-o', label='AlexNet Test Loss', lw=2)
    plt.plot(range(1, epochs + 1), vgg_history['test_loss'], 'r--s', label='VGG16 Test Loss', lw=2)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.title('AlexNet vs VGG16: Loss Progression', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in alex_history['test_acc']], 'b-o', label='AlexNet Test Acc', lw=2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in vgg_history['test_acc']], 'r--s', label='VGG16 Test Acc', lw=2)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('AlexNet vs VGG16: Accuracy Progression', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'transfer_learning_comparison.png'), dpi=300)
    plt.close()
    
    # Confusion matrices side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cm_alex = confusion_matrix(y_true, alex_preds)
    cm_vgg = confusion_matrix(y_true, vgg_preds)
    
    sns.heatmap(cm_alex, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(f"AlexNet (Acc: {alex_acc*100:.2f}%)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Digit')
    axes[0].set_ylabel('Actual Digit')
    
    sns.heatmap(cm_vgg, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False)
    axes[1].set_title(f"VGG16 (Acc: {vgg_acc*100:.2f}%)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Digit')
    axes[1].set_ylabel('Actual Digit')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'transfer_confusion_matrices.png'), dpi=300)
    plt.close()
    
    return results


if __name__ == '__main__':
    run_transfer_experiments(epochs=2, max_train_samples=1500, max_test_samples=800)

