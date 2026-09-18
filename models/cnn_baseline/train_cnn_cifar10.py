import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

# Add project root (~/NSSON) to sys.path so 'nsson_utils' is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from sim.nsson_utils.run_utils import make_run_dir, csv_logger


def get_loaders(data_dir, batch_size):
    # Standard CIFAR-10 stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=os.path.join(data_dir, "cifar10"),
        train=True,
        download=True,
        transform=train_transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=os.path.join(data_dir, "cifar10"),
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    return running_loss / total, correct / total


def main():
    data_dir = os.path.expanduser("~/NSSON/data")
    os.makedirs(data_dir, exist_ok=True)

    # Per-run directory under shared/cnn_runs
    run_root = "~/NSSON/shared/cnn_runs"
    run_dir = make_run_dir(run_root, prefix="cnn_resnet18")
    print("CNN run dir:", run_dir)

    train_log = csv_logger(
        os.path.join(run_dir, "train_log.csv"),
        fieldnames=["epoch", "train_loss", "train_acc"],
    )
    test_log = csv_logger(
        os.path.join(run_dir, "test_log.csv"),
        fieldnames=["epoch", "test_loss", "test_acc"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_loaders(data_dir, batch_size=128)

    # CIFAR-style ResNet-18 for CIFAR-10 (10 classes)
    model = torchvision.models.resnet18(num_classes=10)
    # Replace 7x7 conv + stride 2 + maxpool with 3x3 conv, stride 1, no maxpool
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
    )

    best_acc = 0.0
    num_epochs = 180

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = eval_epoch(
            model, test_loader, criterion, device
        )
        scheduler.step()

        print(
            f"Epoch {epoch+1:03d}/{num_epochs} "
            f"Train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
            f"Test loss {test_loss:.4f} acc {test_acc*100:.2f}%"
        )

        # Log to CSVs
        train_log({
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
        })
        test_log({
            "epoch": epoch + 1,
            "test_loss": f"{test_loss:.6f}",
            "test_acc": f"{test_acc:.6f}",
        })

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_dir = os.path.join(
                os.path.expanduser("~/NSSON/models/cnn_baseline"),
                "checkpoints",
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "resnet18_cifar10_best.pth")
            torch.save(
                {"state_dict": model.state_dict(), "acc": best_acc},
                ckpt_path,
            )

    elapsed = time.time() - start_time
    print(f"Best test accuracy: {best_acc*100:.2f}%")
    print(f"Total training time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
