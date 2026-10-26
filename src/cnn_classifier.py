"""
Part 1: CNN Architecture for MNIST Classification
Lab 2: Deep Learning — FST Tanger
Student: Yassine Kamouss (LSI)
"""

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_mnist_loaders


class MNISTConvNet(nn.Module):
    """
    Standard Deep Convolutional Neural Network for MNIST digit recognition.
    Architecture:
      ConvBlock 1: Conv2d(1->32, k=3, p=1, s=1) -> BatchNorm -> ReLU -> MaxPool2d(2)  [14x14]
      ConvBlock 2: Conv2d(32->64, k=3, p=1, s=1) -> BatchNorm -> ReLU -> MaxPool2d(2)  [7x7]
      ConvBlock 3: Conv2d(64->128, k=3, p=1, s=1) -> BatchNorm -> ReLU -> AdaptiveAvgPool(4x4)
      Classifier: Flatten -> Linear(128*4*4 -> 256) -> ReLU -> Dropout(0.4) -> Linear(256 -> 10)
    """
    def __init__(self, use_dropout: bool = True, use_batchnorm: bool = True, dropout_rate: float = 0.4):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        
        # Block 1
        b1 = [nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)]
        if use_batchnorm:
            b1.append(nn.BatchNorm2d(32))
        b1.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])
        self.block1 = nn.Sequential(*b1)
        
        # Block 2
        b2 = [nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)]
        if use_batchnorm:
            b2.append(nn.BatchNorm2d(64))
        b2.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])
        self.block2 = nn.Sequential(*b2)
        
        # Block 3
        b3 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)]
        if use_batchnorm:
            b3.append(nn.BatchNorm2d(128))
        b3.extend([nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4, 4))])
        self.block3 = nn.Sequential(*b3)
        
        # Classifier
        clf = [nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True)]
        if use_dropout:
            clf.append(nn.Dropout(p=dropout_rate))
        clf.append(nn.Linear(256, 10))
        self.classifier = nn.Sequential(*clf)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer, device):
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


def evaluate(model, loader, criterion, device):
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


def run_cnn_training(epochs: int = 3, lr: float = 0.001, optimizer_name: str = 'adam',
                     weight_decay: float = 1e-4, max_train_samples: int = 20000, max_test_samples: int = 4000,
                     device=None, save_dir: str = './figures'):
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training CNN on Device: {device} | Optimizer: {optimizer_name} | Epochs: {epochs}")
    
    train_loader, test_loader = get_mnist_loaders(batch_size=128, augment=False,
                                                  max_train_samples=max_train_samples,
                                                  max_test_samples=max_test_samples)
    model = MNISTConvNet(use_dropout=True, use_batchnorm=True, dropout_rate=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    start_time = time.time()
    for ep in range(1, epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_f1, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['test_loss'].append(v_loss)
        history['test_acc'].append(v_acc)
        print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {t_loss:.4f} - Acc: {t_acc*100:.2f}% | Test Loss: {v_loss:.4f} - Acc: {v_acc*100:.2f}%")
        
    total_time = time.time() - start_time
    print(f"[+] CNN Training Completed in {total_time:.2f} seconds.")
    
    final_loss, final_acc, final_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    
    # Save training curves
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], 'b-o', label='Train Loss', lw=2)
    plt.plot(range(1, epochs + 1), history['test_loss'], 'r--s', label='Test Loss', lw=2)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Cross-Entropy Loss', fontsize=11)
    plt.title('Custom CNN: Loss vs Epochs', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history['train_acc']], 'b-o', label='Train Acc', lw=2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history['test_acc']], 'r--s', label='Test Acc', lw=2)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Custom CNN: Accuracy vs Epochs', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cnn_training_curves.png'), dpi=300)
    plt.close()
    
    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Custom CNN Confusion Matrix (Acc: {final_acc*100:.2f}%)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Digit', fontsize=11)
    plt.ylabel('Actual Digit', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cnn_confusion_matrix.png'), dpi=300)
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
    run_cnn_training(epochs=5)
