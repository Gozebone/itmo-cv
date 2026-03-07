import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from tqdm.auto import tqdm


COLOR_INDEX = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


class DVMCarsColorDataset(Dataset):
    def __init__(self, root_dir: str, samples: List[Tuple[str, int]], color_idx_to_label: Dict[int, str], transform=None):
        self.root_dir = root_dir
        self.samples = samples
        self.color_idx_to_label = color_idx_to_label
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, target = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def collect_samples(root: str, min_samples_per_class: int) -> Tuple[List[Tuple[str, int]], Dict[str, int], Dict[int, str]]:
    all_files: List[Tuple[str, str, str]] = [] 
    color_freq: Dict[str, int] = {}

    for brand in sorted(os.listdir(root)):
        brand_dir = os.path.join(root, brand)
        if not os.path.isdir(brand_dir):
            continue
        for year in sorted(os.listdir(brand_dir)):
            year_dir = os.path.join(brand_dir, year)
            if not os.path.isdir(year_dir):
                continue
            for fname in os.listdir(year_dir):
                if not is_image_file(fname):
                    continue
                tokens = fname.split("$$")
                if len(tokens) <= COLOR_INDEX:
                    continue
                color = tokens[COLOR_INDEX]
                all_files.append((brand, year, fname))
                color_freq[color] = color_freq.get(color, 0) + 1

    kept_colors = {c for c, cnt in color_freq.items() if cnt >= min_samples_per_class}

    color_to_idx: Dict[str, int] = {}
    samples: List[Tuple[str, int]] = []
    for brand, year, fname in all_files:
        tokens = fname.split("$$")
        color = tokens[COLOR_INDEX]
        if color not in kept_colors:
            continue
        if color not in color_to_idx:
            color_to_idx[color] = len(color_to_idx)
        cls_idx = color_to_idx[color]
        rel_path = os.path.join(brand, year, fname)
        samples.append((rel_path, cls_idx))

    idx_to_color = {v: k for k, v in color_to_idx.items()}
    return samples, color_to_idx, idx_to_color



def build_resnet50_scratch(num_classes: int, device: str):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)
    return model


@dataclass
class TrainConfig:
    data_root: str
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    val_size: float = 0.2
    random_state: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    min_samples_per_class: int = 30
    label_smoothing: float = 0.0


INPUT_SIZE = 224


def get_transforms() -> Tuple[T.Compose, T.Compose]:
    train_tf = T.Compose(
        [
            T.Resize((256, 256)),  # ResNet: resize before crop
            T.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.CenterCrop(INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def make_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    samples, color_to_idx, idx_to_color = collect_samples(cfg.data_root, cfg.min_samples_per_class)
    train_samples, val_samples = train_test_split(
        samples, test_size=cfg.val_size, random_state=cfg.random_state, stratify=[s[1] for s in samples]
    )

    train_tf, val_tf = get_transforms()

    train_ds = DVMCarsColorDataset(cfg.data_root, train_samples, idx_to_color, transform=train_tf)
    val_ds = DVMCarsColorDataset(cfg.data_root, val_samples, idx_to_color, transform=val_tf)
    train_targets = [t for _, t in train_samples]
    class_counts: Dict[int, int] = {}
    for t in train_targets:
        class_counts[t] = class_counts.get(t, 0) + 1
    sample_weights = [1.0 / class_counts[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, idx_to_color


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    total_batches = len(loader)
    loop = tqdm(loader, total=total_batches, desc="Batches", leave=False)
    for imgs, targets in loop:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        loop.set_postfix({"loss": f"{loss.item():.4f}"})
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    correct = 0
    total = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        correct += (preds == targets).sum().item()
        total += targets.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    acc = correct / total if total > 0 else 0.0
    f1_macro = float(f1_score(all_targets, all_preds, average="macro"))
    return acc, f1_macro


def build_pretrained_resnet(num_classes: int, device: str):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)
    return model


def build_pretrained_mobilenet(num_classes: int, device: str):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    last_layer = model.classifier[-1]
    in_features = getattr(last_layer, "in_features", None) or getattr(last_layer, "out_features", None)
    if in_features is None:
        raise RuntimeError("Cannot determine in_features for MobileNet classifier replacement")
    in_features = int(in_features)
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.to(device)
    return model


def build_resnet18_scratch(num_classes: int, device: str):
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes=1000):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.to(device)
    return model


