"""
scene_recog_cnn.py

Main script for scene recognition using CNN-based transfer learning.
Trains a ConvNeXt Small and ResNet152 ensemble for 15-class scene classification.

Usage (command line):
    # Training
    python3 scene_recog_cnn.py --phase train --train_data_dir ./data/train --model_dir .

    # Testing
    python3 scene_recog_cnn.py --phase test --test_data_dir ./data/test --model_dir .

Usage (Python):
    from scene_recog_cnn import train, test
    train("./data/train", ".")
    test("./data/test", ".")
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from model import get_convnext_small, get_resnet152


# ─── Label mapping (DO NOT CHANGE) ────────────────────────────────────────────
LABEL_MAP = {
    "bedroom": 1, "Coast": 2, "Forest": 3, "Highway": 4,
    "industrial": 5, "Insidecity": 6, "kitchen": 7, "livingroom": 8,
    "Mountain": 9, "Office": 10, "OpenCountry": 11, "store": 12,
    "Street": 13, "Suburb": 14, "TallBuilding": 15,
}
NUM_CLASSES = 15
IMG_SIZE    = 224


# ─── Subroutines ──────────────────────────────────────────────────────────────

def set_seed(seed=42):
    """Sets random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SceneDataset(Dataset):
    """
    Custom PyTorch Dataset for scene recognition.

    Args:
        paths     (list): List of image file paths.
        labels    (list): Corresponding list of integer labels (0-indexed).
        transform:        torchvision transforms to apply to each image.
    """
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_data(data_dir):
    """
    Loads image paths and labels from a directory structured by class subfolders.

    Args:
        data_dir (str): Root directory containing one subfolder per class.

    Returns:
        paths  (list): List of image file paths.
        labels (list): List of 0-indexed integer labels.
    """
    paths, labels = [], []
    for class_name, label in LABEL_MAP.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Missing class folder: {class_dir}")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label - 1)
    return paths, labels


def get_accuracy(loader, model, device):
    """
    Computes accuracy of a single model on a given DataLoader.

    Args:
        loader (DataLoader): DataLoader to evaluate on.
        model  (nn.Module):  Model to evaluate.
        device:              Device to run inference on.

    Returns:
        accuracy (float): Accuracy as a fraction between 0 and 1.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def get_ensemble_accuracy(loader, convnext_model, resnet_model, device):
    """
    Computes soft-voting ensemble accuracy on a given DataLoader.
    Averages softmax probabilities of ConvNeXt Small and ResNet152.

    Args:
        loader         (DataLoader): DataLoader to evaluate on.
        convnext_model (nn.Module):  ConvNeXt Small model.
        resnet_model   (nn.Module):  ResNet152 model.
        device:                      Device to run inference on.

    Returns:
        accuracy   (float):     Accuracy as a fraction between 0 and 1.
        all_preds  (np.array):  1-indexed predicted labels.
        all_labels (np.array):  1-indexed true labels.
    """
    convnext_model.eval()
    resnet_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            prob_convnext = torch.softmax(convnext_model(imgs), dim=1)
            prob_resnet   = torch.softmax(resnet_model(imgs),   dim=1)
            avg_probs     = (prob_convnext + prob_resnet) / 2.0
            preds         = avg_probs.argmax(dim=1)
            all_preds.extend((preds + 1).cpu().numpy())
            all_labels.extend((labels + 1).numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = np.mean(all_preds == all_labels)
    return accuracy, all_preds, all_labels


def save_model(model, save_path):
    """
    Saves model state dict to the specified path.

    Args:
        model     (nn.Module): Model to save.
        save_path (str):       Full path to save the weights file.
    """
    torch.save(model.state_dict(), save_path)
    print(f"  ✓ Saved model to: {save_path}")


def train_one_model(model, train_loader, val_loader, save_path,
                    lr=1e-4, epochs=50, patience=7,
                    label_smoothing=0.1, device="cpu", seed=42):
    """
    Trains a single model with early stopping and ReduceLROnPlateau scheduler.

    Args:
        model           (nn.Module):  Model to train.
        train_loader    (DataLoader): Training DataLoader.
        val_loader      (DataLoader): Validation DataLoader.
        save_path       (str):        Path to save best model weights.
        lr              (float):      Initial learning rate. Default: 1e-4.
        epochs          (int):        Maximum number of epochs. Default: 50.
        patience        (int):        Early stopping patience. Default: 7.
        label_smoothing (float):      Label smoothing factor. Default: 0.1.
        device:                       Device to train on.
        seed            (int):        Random seed. Default: 42.

    Returns:
        best_val_acc (float): Best validation accuracy achieved during training.
    """
    set_seed(seed)
    criterion        = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer        = optim.Adam(model.parameters(), lr=lr)
    scheduler        = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    best_val_acc     = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total   += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs     = model(imgs)
                loss        = criterion(outputs, labels)
                val_loss    += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        train_acc    = train_correct / train_total
        val_acc      = val_correct   / val_total
        val_loss_avg = val_loss / len(val_loader)

        scheduler.step(val_loss_avg)
        print(f"  Epoch {epoch+1:02d} | "
              f"Train: {train_acc*100:.2f}% | "
              f"Val: {val_acc*100:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Best Val Accuracy: {best_val_acc*100:.2f}%")
    return best_val_acc


# ─── Main train and test functions ────────────────────────────────────────────

def train(train_data_dir, model_dir, **kwargs):
    """
    Main training function. Trains ConvNeXt Small and ResNet152 sequentially
    and saves both model weights to model_dir.

    Training setup:
        - ConvNeXt Small: all layers unfrozen, hflip augmentation, label smoothing=0.1
        - ResNet152: all layers unfrozen, hflip + brightness augmentation, label smoothing=0.1
        - Optimiser: Adam, lr=1e-4
        - Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
        - Early stopping: patience=7

    Arguments:
        train_data_dir (str): The directory of training data.
        model_dir      (str): The directory to save trained model weights.
                              Saves: trained_cnn.pth (ConvNeXt), trained_cnn_res.pth (ResNet152)
        **kwargs (optional):  Other kwargs with default values:
            val_split   (float): Validation split fraction. Default: 0.2.
            batch_size  (int):   Batch size. Default: 32.
            lr          (float): Learning rate. Default: 1e-4.
            epochs      (int):   Max epochs. Default: 50.
            patience    (int):   Early stopping patience. Default: 7.
            num_workers (int):   DataLoader workers. Default: 4.
            seed        (int):   Random seed. Default: 42.

    Return:
        train_accuracy (float): Ensemble training accuracy.
    """
    val_split   = kwargs.get("val_split",   0.2)
    batch_size  = kwargs.get("batch_size",  32)
    lr          = kwargs.get("lr",          1e-4)
    epochs      = kwargs.get("epochs",      50)
    patience    = kwargs.get("patience",    7)
    num_workers = kwargs.get("num_workers", 4)
    seed        = kwargs.get("seed",        42)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(model_dir, exist_ok=True)

    # ── Load and split data ───────────────────────────────────────────────────
    all_paths, all_labels = load_data(train_data_dir)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=val_split, stratify=all_labels, random_state=seed
    )
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    # ── Transforms ───────────────────────────────────────────────────────────
    convnext_train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    resnet_train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_ds     = SceneDataset(val_paths, val_labels, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ── Train ConvNeXt Small ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Training ConvNeXt Small...")
    print("="*60)
    set_seed(seed)
    convnext_model = get_convnext_small(pretrained=True)
    for param in convnext_model.parameters():
        param.requires_grad = True
    convnext_model = convnext_model.to(device)

    convnext_train_ds     = SceneDataset(train_paths, train_labels, transform=convnext_train_transform)
    convnext_train_loader = DataLoader(convnext_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    convnext_save_path    = os.path.join(model_dir, "trained_cnn.pth")

    train_one_model(convnext_model, convnext_train_loader, val_loader,
                    save_path=convnext_save_path,
                    lr=lr, epochs=epochs, patience=patience,
                    device=device, seed=seed)

    # ── Train ResNet152 ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Training ResNet152...")
    print("="*60)
    set_seed(seed)
    resnet_model = get_resnet152(pretrained=True)
    for param in resnet_model.parameters():
        param.requires_grad = True
    resnet_model = resnet_model.to(device)

    resnet_train_ds     = SceneDataset(train_paths, train_labels, transform=resnet_train_transform)
    resnet_train_loader = DataLoader(resnet_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    resnet_save_path    = os.path.join(model_dir, "trained_cnn_res.pth")

    train_one_model(resnet_model, resnet_train_loader, val_loader,
                    save_path=resnet_save_path,
                    lr=lr, epochs=epochs, patience=patience,
                    device=device, seed=seed)

    # ── Compute ensemble train accuracy ───────────────────────────────────────
    print("\n" + "="*60)
    print("Computing ensemble training accuracy...")
    print("="*60)
    convnext_model.load_state_dict(torch.load(convnext_save_path, map_location=device))
    resnet_model.load_state_dict(torch.load(resnet_save_path,    map_location=device))

    full_train_ds     = SceneDataset(train_paths, train_labels, transform=val_transform)
    full_train_loader = DataLoader(full_train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_accuracy, _, _ = get_ensemble_accuracy(full_train_loader, convnext_model, resnet_model, device)
    print(f"Ensemble Train Accuracy: {train_accuracy*100:.2f}%")

    return train_accuracy


def test(test_data_dir, model_dir, **kwargs):
    """
    Main testing function. Loads ConvNeXt Small and ResNet152 from model_dir
    and runs soft-voting ensemble on the test data.

    Ensemble method: Averages softmax probabilities of both models
    and takes the argmax as the final prediction.

    Arguments:
        test_data_dir (str): The directory of test data.
                             Same folder structure as train_data_dir.
        model_dir     (str): The directory containing trained model weights.
                             Expected files: trained_cnn.pth, trained_cnn_res.pth
        **kwargs (optional): Other kwargs with default values:
            batch_size  (int): Batch size. Default: 32.
            num_workers (int): DataLoader workers. Default: 4.

    Return:
        test_accuracy (float): The ensemble test accuracy.
    """
    batch_size  = kwargs.get("batch_size",  32)
    num_workers = kwargs.get("num_workers", 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load test data ────────────────────────────────────────────────────────
    test_paths, test_labels = load_data(test_data_dir)
    print(f"Test images: {len(test_paths)}")

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_ds     = SceneDataset(test_paths, test_labels, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ── Load ConvNeXt Small ───────────────────────────────────────────────────
    convnext_path  = os.path.join(model_dir, "trained_cnn.pth")
    convnext_model = get_convnext_small(pretrained=False)
    convnext_model.load_state_dict(torch.load(convnext_path, map_location=device))
    convnext_model = convnext_model.to(device)
    print(f"✓ ConvNeXt Small loaded from: {convnext_path}")

    # ── Load ResNet152 ────────────────────────────────────────────────────────
    resnet_path  = os.path.join(model_dir, "trained_cnn_res.pth")
    resnet_model = get_resnet152(pretrained=False)
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet_model = resnet_model.to(device)
    print(f"✓ ResNet152 loaded from: {resnet_path}")

    # ── Ensemble inference ────────────────────────────────────────────────────
    test_accuracy, all_preds, all_labels = get_ensemble_accuracy(
        test_loader, convnext_model, resnet_model, device
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\nEnsemble Test Accuracy: {test_accuracy*100:.2f}%")

    inv_map = {v: k for k, v in LABEL_MAP.items()}
    print("\nPer-class accuracy:")
    for label_val in range(1, NUM_CLASSES + 1):
        mask = all_labels == label_val
        if mask.sum() == 0:
            continue
        class_acc = np.mean(all_preds[mask] == all_labels[mask])
        print(f"  {inv_map[label_val]:<15}: {class_acc*100:.1f}%  ({mask.sum()} imgs)")

    return test_accuracy


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train', 'test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir',  default='./data/test/',  help='the directory of testing data')
    parser.add_argument('--model_dir',      default='.',       help='the directory of trained models')
    opt = parser.parse_args()

    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)
