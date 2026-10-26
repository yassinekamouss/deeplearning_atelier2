"""
Dataset and DataLoader utilities for MNIST and Object Detection (Faster R-CNN).
Lab 2: Deep Learning — FST Tanger
Student: Yassine Kamouss (LSI)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np


def get_mnist_transforms(augment: bool = False, three_channels: bool = False, resize: int = None):
    """
    Returns standard MNIST transformations.
    """
    transform_list = []
    
    if resize is not None:
        transform_list.append(transforms.Resize((resize, resize)))
        
    if augment:
        transform_list.extend([
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop(28 if resize is None else resize, padding=2, padding_mode='edge'),
        ])
        
    transform_list.append(transforms.ToTensor())
    
    if three_channels:
        # Replicate grayscale channel to 3 channels for models like AlexNet / VGG16
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        transform_list.append(transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)))
    else:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
    return transforms.Compose(transform_list)


def get_mnist_loaders(root: str = './data', batch_size: int = 128, augment: bool = False,
                      three_channels: bool = False, resize: int = None, num_workers: int = 0,
                      max_train_samples: int = None, max_test_samples: int = None):
    """
    Build standard training and testing DataLoaders for MNIST with CPU optimization.
    """
    train_transform = get_mnist_transforms(augment=augment, three_channels=three_channels, resize=resize)
    test_transform = get_mnist_transforms(augment=False, three_channels=three_channels, resize=resize)
    
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=test_transform)
    
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:max_train_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        
    if max_test_samples is not None and max_test_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:max_test_samples]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)
    
    return train_loader, test_loader


class MNISTDetectionDataset(Dataset):
    """
    MNIST wrapper formatted for Object Detection with Faster R-CNN.
    Generates bounding box coordinates [xmin, ymin, xmax, ymax]
    around the digit foreground pixels.
    Classes: 1 to 10 (labels 0..9 shifted by +1, where 0 is background).
    """
    def __init__(self, root: str = './data', train: bool = True, size: int = 56):
        super().__init__()
        self.mnist = datasets.MNIST(root=root, train=train, download=True)
        self.size = size
        
    def __len__(self):
        return len(self.mnist)
        
    def __getitem__(self, idx):
        img_pil, label = self.mnist[idx]
        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        
        # Calculate bounding box from foreground pixels (> 0.1 threshold)
        y_indices, x_indices = np.where(img_np > 0.1)
        if len(x_indices) > 0 and len(y_indices) > 0:
            xmin = float(x_indices.min())
            xmax = float(x_indices.max() + 1)
            ymin = float(y_indices.min())
            ymax = float(y_indices.max() + 1)
            # Ensure minimum box dimension
            if xmax - xmin < 2:
                xmax = min(28.0, xmin + 2)
            if ymax - ymin < 2:
                ymax = min(28.0, ymin + 2)
        else:
            xmin, ymin, xmax, ymax = 2.0, 2.0, 26.0, 26.0
            
        # Place or scale to canvas size if requested
        if self.size != 28:
            # Scale coordinates
            scale = self.size / 28.0
            img_pil_resized = img_pil.resize((self.size, self.size))
            img_tensor = transforms.ToTensor()(img_pil_resized)
            # Faster R-CNN expects 3 channels
            img_tensor = img_tensor.repeat(3, 1, 1)
            boxes = torch.as_tensor([[xmin * scale, ymin * scale, xmax * scale, ymax * scale]], dtype=torch.float32)
        else:
            img_tensor = transforms.ToTensor()(img_pil).repeat(3, 1, 1)
            boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
            
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.as_tensor([label + 1], dtype=torch.int64)  # 0 is background
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((1,), dtype=torch.int64)
        }
        
        return img_tensor, target


def detection_collate_fn(batch):
    """
    Collate function for Faster R-CNN tuple inputs.
    """
    return tuple(zip(*batch))


def get_detection_loaders(root: str = './data', batch_size: int = 16, size: int = 56, num_workers: int = 0,
                          max_train_samples: int = None, max_test_samples: int = None):
    """
    Build training and testing DataLoaders for Faster R-CNN.
    """
    train_ds = MNISTDetectionDataset(root=root, train=True, size=size)
    test_ds = MNISTDetectionDataset(root=root, train=False, size=size)
    
    if max_train_samples is not None and max_train_samples < len(train_ds):
        train_ds = torch.utils.data.Subset(train_ds, range(max_train_samples))
    if max_test_samples is not None and max_test_samples < len(test_ds):
        test_ds = torch.utils.data.Subset(test_ds, range(max_test_samples))
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=detection_collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=detection_collate_fn, num_workers=num_workers)
    
    return train_loader, test_loader
