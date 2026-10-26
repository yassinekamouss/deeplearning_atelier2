"""
Master Experiment Runner & Comparative Benchmark for Lab 2
Integrates: Custom CNN, Faster R-CNN, AlexNet, VGG16, and Vision Transformer (ViT)
Lab 2: Deep Learning — FST Tanger
Student: Yassine Kamouss (LSI)
"""

import os
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modular components
from data_utils import get_mnist_loaders, get_detection_loaders, get_mnist_transforms
from cnn_classifier import MNISTConvNet, evaluate as evaluate_cnn
from faster_rcnn_detector import get_faster_rcnn_model, evaluate_detection, visualize_predictions
from transfer_learning import get_finetuned_alexnet, get_finetuned_vgg16, evaluate_transfer
from vit_scratch import MyViT, evaluate_vit, visualize_vit_attention


def measure_inference_latency(model, sample_input, device, num_runs: int = 50):
    """
    Measure average inference latency in milliseconds per sample.
    """
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(5):
            _ = model(sample_input)
            
        start = time.time()
        for _ in range(num_runs):
            _ = model(sample_input)
        elapsed = (time.time() - start) / num_runs * 1000.0  # ms
    return elapsed


def generate_mnist_gallery(save_path: str = './figures/mnist_dataset_samples.png'):
    """
    Plots a sample grid of MNIST digits with labels.
    """
    ds = datasets.MNIST(root='./data', train=True, download=True)
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
    axes = axes.flatten()
    
    seen = {}
    idx = 0
    while len(seen) < 10 and idx < len(ds):
        img, lbl = ds[idx]
        if lbl not in seen:
            seen[lbl] = img
        idx += 1
        
    for digit in range(10):
        ax = axes[digit]
        ax.imshow(seen[digit], cmap='gray')
        ax.set_title(f"Class: {digit}", fontsize=11, fontweight='bold')
        ax.axis('off')
        
    plt.suptitle("MNIST Dataset Samples (Classes 0 to 9)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_all_benchmarks(save_dir: str = './figures', fast_mode: bool = False):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f" Lab 2 Master Benchmark Suite — Device: {device} ")
    print(f"==================================================")
    
    generate_mnist_gallery(os.path.join(save_dir, 'mnist_dataset_samples.png'))
    benchmark_records = []
    
    # Common test loaders
    max_test = 2000 if fast_mode else None
    _, test_loader_1ch = get_mnist_loaders(batch_size=128, augment=False, max_test_samples=max_test)
    _, test_loader_3ch = get_mnist_loaders(batch_size=128, augment=False, three_channels=True, resize=64, max_test_samples=max_test)
    
    # -------------------------------------------------------------------------
    # 1. Custom CNN
    # -------------------------------------------------------------------------
    print("\n[1/5] Training & Evaluating Custom CNN...")
    from cnn_classifier import run_cnn_training
    cnn_epochs = 2 if fast_mode else 4
    cnn_res = run_cnn_training(epochs=cnn_epochs, device=device, save_dir=save_dir)
    
    # Latency test
    sample_1ch = torch.randn(1, 1, 28, 28).to(device)
    cnn_latency = measure_inference_latency(cnn_res['model'], sample_1ch, device)
    
    benchmark_records.append({
        'Model': 'Custom CNN (Baseline + Reg)',
        'Architecture Type': 'Feed-Forward ConvNet',
        'Accuracy (%)': round(cnn_res['accuracy'] * 100, 2),
        'F1-Score': round(cnn_res['f1_score'], 4),
        'Test Loss': round(cnn_res['test_loss'], 4),
        'Training Time (s)': round(cnn_res['training_time'], 1),
        'Inference Latency (ms)': round(cnn_latency, 2),
        'Parameters': cnn_res['parameters']
    })
    
    # -------------------------------------------------------------------------
    # 2. Faster R-CNN
    # -------------------------------------------------------------------------
    print("\n[2/5] Training & Evaluating Faster R-CNN Detection Architecture...")
    from faster_rcnn_detector import run_faster_rcnn_training
    rcnn_batches = 40 if fast_mode else 80
    rcnn_res = run_faster_rcnn_training(num_batches=rcnn_batches, batch_size=16, device=device, save_dir=save_dir)
    
    sample_img_rcnn = [torch.randn(3, 56, 56).to(device)]
    rcnn_latency = measure_inference_latency(rcnn_res['model'], sample_img_rcnn, device, num_runs=20)
    
    benchmark_records.append({
        'Model': 'Faster R-CNN (MobileNetV3 FPN)',
        'Architecture Type': 'Two-Stage Object Detector',
        'Accuracy (%)': round(rcnn_res['accuracy'] * 100, 2),
        'F1-Score': round(rcnn_res['f1_score'], 4),
        'Test Loss': round(rcnn_res['test_loss'], 4),
        'Training Time (s)': round(rcnn_res['training_time'], 1),
        'Inference Latency (ms)': round(rcnn_latency, 2),
        'Parameters': rcnn_res['parameters']
    })
    
    # -------------------------------------------------------------------------
    # 3. AlexNet (Fine-Tuned Transfer Learning)
    # -------------------------------------------------------------------------
    print("\n[3/5] Fine-Tuning AlexNet...")
    alex_epochs = 1 if fast_mode else 2
    alexnet = get_finetuned_alexnet(num_classes=10, freeze_backbone=True).to(device)
    crit = torch.nn.CrossEntropyLoss()
    opt_alex = torch.optim.Adam(filter(lambda p: p.requires_grad, alexnet.parameters()), lr=0.001)
    
    train_loader_3ch, _ = get_mnist_loaders(batch_size=128, augment=False, three_channels=True, resize=64)
    start_alex = time.time()
    for ep in range(1, alex_epochs + 1):
        for b_idx, (imgs, lbls) in enumerate(train_loader_3ch):
            if fast_mode and b_idx >= 100:
                break
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt_alex.zero_grad()
            out = alexnet(imgs)
            l = crit(out, lbls)
            l.backward()
            opt_alex.step()
    alex_time = time.time() - start_alex
    alex_loss, alex_acc, alex_f1, _, _ = evaluate_transfer(alexnet, test_loader_3ch, crit, device)
    
    sample_3ch = torch.randn(1, 3, 64, 64).to(device)
    alex_latency = measure_inference_latency(alexnet, sample_3ch, device)
    
    benchmark_records.append({
        'Model': 'AlexNet (Fine-Tuned Pretrained)',
        'Architecture Type': 'Transfer Learning CNN',
        'Accuracy (%)': round(alex_acc * 100, 2),
        'F1-Score': round(alex_f1, 4),
        'Test Loss': round(alex_loss, 4),
        'Training Time (s)': round(alex_time, 1),
        'Inference Latency (ms)': round(alex_latency, 2),
        'Parameters': sum(p.numel() for p in alexnet.parameters() if p.requires_grad)
    })
    
    # -------------------------------------------------------------------------
    # 4. VGG16 (Fine-Tuned Transfer Learning)
    # -------------------------------------------------------------------------
    print("\n[4/5] Fine-Tuning VGG16...")
    vgg_epochs = 1
    vgg16 = get_finetuned_vgg16(num_classes=10, freeze_backbone=True).to(device)
    opt_vgg = torch.optim.Adam(filter(lambda p: p.requires_grad, vgg16.parameters()), lr=0.0005)
    
    start_vgg = time.time()
    for ep in range(1, vgg_epochs + 1):
        for b_idx, (imgs, lbls) in enumerate(train_loader_3ch):
            if b_idx >= (35 if fast_mode else 120):
                break
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt_vgg.zero_grad()
            out = vgg16(imgs)
            l = crit(out, lbls)
            l.backward()
            opt_vgg.step()
    vgg_time = time.time() - start_vgg
    vgg_loss, vgg_acc, vgg_f1, _, _ = evaluate_transfer(vgg16, test_loader_3ch, crit, device)
    vgg_latency = measure_inference_latency(vgg16, sample_3ch, device)
    
    benchmark_records.append({
        'Model': 'VGG16 (Fine-Tuned Pretrained)',
        'Architecture Type': 'Deep Transfer Learning CNN',
        'Accuracy (%)': round(vgg_acc * 100, 2),
        'F1-Score': round(vgg_f1, 4),
        'Test Loss': round(vgg_loss, 4),
        'Training Time (s)': round(vgg_time, 1),
        'Inference Latency (ms)': round(vgg_latency, 2),
        'Parameters': sum(p.numel() for p in vgg16.parameters() if p.requires_grad)
    })
    
    # -------------------------------------------------------------------------
    # 5. Vision Transformer (ViT from Scratch)
    # -------------------------------------------------------------------------
    print("\n[5/5] Training Vision Transformer (ViT) From Scratch...")
    from vit_scratch import run_vit_training
    vit_epochs = 2 if fast_mode else 4
    vit_res = run_vit_training(epochs=vit_epochs, device=device, save_dir=save_dir)
    vit_latency = measure_inference_latency(vit_res['model'], sample_1ch, device)
    
    benchmark_records.append({
        'Model': 'Vision Transformer (ViT Scratch)',
        'Architecture Type': 'Pure Attention / Transformer',
        'Accuracy (%)': round(vit_res['accuracy'] * 100, 2),
        'F1-Score': round(vit_res['f1_score'], 4),
        'Test Loss': round(vit_res['test_loss'], 4),
        'Training Time (s)': round(vit_res['training_time'], 1),
        'Inference Latency (ms)': round(vit_latency, 2),
        'Parameters': vit_res['parameters']
    })
    
    # -------------------------------------------------------------------------
    # Summary Tables & Plots
    # -------------------------------------------------------------------------
    df = pd.DataFrame(benchmark_records)
    csv_path = os.path.join(save_dir, 'benchmark_comparison.csv')
    df.to_csv(csv_path, index=False)
    print("\n================ BENCHMARK RESULTS ================")
    print(df.to_string(index=False))
    
    # Bar Chart Comparisons
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Accuracy
    axes[0, 0].bar(df['Model'], df['Accuracy (%)'], color=colors, edgecolor='black', alpha=0.85)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Classification Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticklabels(df['Model'], rotation=25, ha='right', fontsize=9)
    for i, v in enumerate(df['Accuracy (%)']):
        axes[0, 0].text(i, v + 0.8, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9)
    axes[0, 0].set_ylim(0, 105)
    
    # 2. F1 Score
    axes[0, 1].bar(df['Model'], df['F1-Score'], color=colors, edgecolor='black', alpha=0.85)
    axes[0, 1].set_ylabel('Macro F1-Score', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Macro F1-Score Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticklabels(df['Model'], rotation=25, ha='right', fontsize=9)
    for i, v in enumerate(df['F1-Score']):
        axes[0, 1].text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold', fontsize=9)
    axes[0, 1].set_ylim(0, 1.1)
    
    # 3. Training Time
    axes[1, 0].bar(df['Model'], df['Training Time (s)'], color=colors, edgecolor='black', alpha=0.85)
    axes[1, 0].set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Training Computational Cost', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticklabels(df['Model'], rotation=25, ha='right', fontsize=9)
    for i, v in enumerate(df['Training Time (s)']):
        axes[1, 0].text(i, v + 2, f"{v:.1f}s", ha='center', fontweight='bold', fontsize=9)
        
    # 4. Latency
    axes[1, 1].bar(df['Model'], df['Inference Latency (ms)'], color=colors, edgecolor='black', alpha=0.85)
    axes[1, 1].set_ylabel('Latency (ms / sample)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Inference Latency (CPU Throughput)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticklabels(df['Model'], rotation=25, ha='right', fontsize=9)
    for i, v in enumerate(df['Inference Latency (ms)']):
        axes[1, 1].text(i, v + 0.2, f"{v:.1f}ms", ha='center', fontweight='bold', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_metrics_comparison_barcharts.png'), dpi=300)
    plt.close()
    
    # Parameter vs Accuracy Scatter Plot
    plt.figure(figsize=(9, 6))
    for i, row in df.iterrows():
        plt.scatter(row['Parameters'], row['Accuracy (%)'], s=180, color=colors[i], edgecolors='black', label=row['Model'])
        plt.annotate(row['Model'].split(' ')[0], (row['Parameters'], row['Accuracy (%)']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
        
    plt.xscale('log')
    plt.xlabel('Trainable Parameters (Log Scale)', fontsize=11, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Parameter Efficiency vs Classification Accuracy', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'architecture_tradeoff_scatter.png'), dpi=300)
    plt.close()
    
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run Lab 2 Computer Vision Benchmarks")
    parser.add_argument('--full', action='store_true', help="Run full-scale epochs (slower on CPU)")
    args = parser.parse_args()
    
    run_all_benchmarks(fast_mode=not args.full)

