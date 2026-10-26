"""
Part 1: Faster R-CNN for MNIST Digit Detection & Classification
Lab 2: Deep Learning — FST Tanger
Student: Yassine Kamouss (LSI)
"""

import time
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import MNISTDetectionDataset, detection_collate_fn


def get_faster_rcnn_model(num_classes: int = 11, backbone: str = 'mobilenet', img_size: int = 56):
    """
    Build Faster R-CNN model adapted for MNIST digits (10 digit classes + 1 background).
    min_size and max_size are set to match image dimensions to avoid default 800x800 upsampling.
    """
    if backbone == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT', min_size=img_size, max_size=img_size)
    else:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT', min_size=img_size, max_size=img_size)
        
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    return model


def compute_iou(box1, box2):
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_detection(model, loader, device, max_samples: int = 1000):
    model.eval()
    all_preds, all_labels = [], []
    ious = []
    
    samples_processed = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for out, tgt in zip(outputs, targets):
                gt_label = tgt['labels'][0].item() - 1
                gt_box = tgt['boxes'][0].cpu().numpy()
                
                if len(out['boxes']) > 0:
                    best_idx = out['scores'].argmax().item()
                    pred_label = out['labels'][best_idx].item() - 1
                    pred_box = out['boxes'][best_idx].cpu().numpy()
                    iou = compute_iou(pred_box, gt_box)
                else:
                    pred_label = 0
                    iou = 0.0
                    
                all_preds.append(pred_label)
                all_labels.append(gt_label)
                ious.append(iou)
                
            samples_processed += len(images)
            if samples_processed >= max_samples:
                break
                
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    mean_iou = np.mean(ious)
    return acc, f1, mean_iou


def visualize_predictions(model, dataset, device, save_path: str = './figures/faster_rcnn_predictions.png'):
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(8):
            img_tensor, tgt = dataset[i * 15]
            gt_box = tgt['boxes'][0].numpy()
            gt_label = tgt['labels'][0].item() - 1
            
            outputs = model([img_tensor.to(device)])
            out = outputs[0]
            
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            ax = axes[i]
            ax.imshow(img_np)
            
            # Ground truth box (Green dashed)
            gt_rect = patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2] - gt_box[0], gt_box[3] - gt_box[1],
                                        linewidth=2, edgecolor='lime', facecolor='none', linestyle='--', label='Ground Truth')
            ax.add_patch(gt_rect)
            
            # Predicted box (Red solid)
            if len(out['boxes']) > 0:
                best_idx = out['scores'].argmax().item()
                pred_box = out['boxes'][best_idx].cpu().numpy()
                pred_label = out['labels'][best_idx].item() - 1
                score = out['scores'][best_idx].item()
                
                pred_rect = patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1],
                                              linewidth=2, edgecolor='red', facecolor='none', label='Prediction')
                ax.add_patch(pred_rect)
                title = f"True: {gt_label} | Pred: {pred_label} ({score:.2f})"
            else:
                title = f"True: {gt_label} | No Det"
                
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
                
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_faster_rcnn_training(num_batches: int = 40, batch_size: int = 16, lr: float = 0.001,
                             max_eval_samples: int = 150, device=None, save_dir: str = './figures'):
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training Faster R-CNN on Device: {device} | Iterations: {num_batches} batches")
    
    train_ds = MNISTDetectionDataset(root='./data', train=True, size=56)
    test_ds = MNISTDetectionDataset(root='./data', train=False, size=56)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                               collate_fn=detection_collate_fn, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                              collate_fn=detection_collate_fn, num_workers=0)
    
    model = get_faster_rcnn_model(num_classes=11, backbone='mobilenet').to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    
    start_time = time.time()
    model.train()
    batch_losses = []
    
    for i, (images, targets) in enumerate(train_loader):
        if i >= num_batches:
            break
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        batch_losses.append(losses.item())
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"  Batch [{i+1:03d}/{num_batches:03d}] | Total Loss: {losses.item():.4f} "
                  f"(Cls: {loss_dict['loss_classifier'].item():.3f}, Box: {loss_dict['loss_box_reg'].item():.3f})")
            
    total_time = time.time() - start_time
    print(f"[+] Faster R-CNN Training Step Completed in {total_time:.2f} seconds.")
    
    # Evaluate
    print("[*] Evaluating Faster R-CNN on test partition...")
    acc, f1, mean_iou = evaluate_detection(model, test_loader, device, max_samples=max_eval_samples)
    print(f"  Test Accuracy: {acc*100:.2f}% | Macro F1: {f1:.4f} | Mean BBox IoU: {mean_iou:.4f}")
    
    # Visualizations
    visualize_predictions(model, test_ds, device, save_path=os.path.join(save_dir, 'faster_rcnn_predictions.png'))
    
    plt.figure(figsize=(8, 4.5))
    plt.plot(batch_losses, 'm-', lw=2, label='Detection Loss (RPN + RoI Head)')
    plt.xlabel('Training Iteration', fontsize=11)
    plt.ylabel('Multi-Task Loss', fontsize=11)
    plt.title('Faster R-CNN Multi-Task Loss Convergence on MNIST', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'faster_rcnn_loss_curve.png'), dpi=300)
    plt.close()
    
    return {
        'model': model,
        'accuracy': acc,
        'f1_score': f1,
        'mean_iou': mean_iou,
        'training_time': total_time,
        'test_loss': np.mean(batch_losses[-10:]),
        'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


if __name__ == '__main__':
    run_faster_rcnn_training(num_batches=60, batch_size=16)
