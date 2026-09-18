import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def build_cnn_model():
    """CIFAR-style ResNet-18 for CIFAR-10 (10 classes)."""
    model = torchvision.models.resnet18(num_classes=10)
    # Adapt conv1 and maxpool for CIFAR-10 32x32 images
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def load_cnn_checkpoint(
    ckpt_path="models/cnn_baseline/checkpoints/resnet18_cifar10_best.pth",
    device=None,
):
    """Load best ResNet-18 CIFAR-10 checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_cnn_model().to(device)

    assert os.path.isfile(ckpt_path), f"Missing CNN checkpoint: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, device


def get_cifar10_test_loader(data_root="data/cifar10", batch_size=128):
    """Return CIFAR-10 test loader with standard transforms."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loader


def evaluate_cnn(model, loader, device):
    """Evaluate CNN accuracy on CIFAR-10 test set."""
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0
