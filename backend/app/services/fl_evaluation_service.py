"""
FL Evaluation Service
Handles pre/post FL model evaluation on public test dataset
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = 128
NUM_CLASSES = 2
CLASSES = ['ALL (Leukemia)', 'Hem (Healthy)']  # Class 0, Class 1 - matches ImageFolder alphabetical order (all, hem)

def build_model(architecture: str) -> nn.Module:
    """Build model architecture matching client's model type"""
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif architecture == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    elif architecture == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model

def load_model_weights(model: nn.Module, model_path: str, device: torch.device) -> nn.Module:
    """Safely load model weights from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model

def get_test_loader(dataset_path: str, batch_size: int = 32) -> DataLoader:
    """
    Load public test dataset (no splitting needed - already separated)

    Args:
        dataset_path: Path to public test dataset folder
        batch_size: Batch size for data loader

    Returns:
        DataLoader for test dataset
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    test_loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    return test_loader

def calculate_essential_metrics(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    """
    Calculate comprehensive evaluation metrics

    Returns:
        Dictionary containing accuracy, precision, recall, F1, ROC-AUC, specificity,
        per-class accuracy, confusion matrix, and sample count
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, pos_label=1, zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, pos_label=1, zero_division=0)),
        'f1_score': float(f1_score(all_labels, all_preds, pos_label=1, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'roc_auc': float(roc_auc_score(all_labels, all_probs[:, 1])) if len(np.unique(all_labels)) > 1 else 0.0,
        'confusion_matrix': cm.tolist(),
        'num_test_samples': len(all_labels)
    }

    # Per-class accuracy
    for i, class_name in enumerate(CLASSES):
        class_mask = (all_labels == i)
        if class_mask.sum() > 0:
            class_acc = accuracy_score(all_labels[class_mask], all_preds[class_mask])
            key = 'healthy_accuracy' if i == 0 else 'leukemia_accuracy'
            metrics[key] = float(class_acc)
        else:
            key = 'healthy_accuracy' if i == 0 else 'leukemia_accuracy'
            metrics[key] = 0.0

    return metrics

def evaluate_client(
    client_id: int,
    client_name: str,
    model_type: str,
    model_path: str,
    public_test_path: str,
    device: torch.device
) -> Dict:
    """
    Evaluate a single client's model on public test dataset

    Args:
        client_id: Client database ID
        client_name: Client name
        model_type: Model architecture (resnet18, mobilenet_v2, etc.)
        model_path: Path to model checkpoint
        public_test_path: Path to public test dataset
        device: PyTorch device (cpu/cuda)

    Returns:
        Dictionary with evaluation metrics and metadata
    """
    logger.info(f"Evaluating {client_name} ({model_type})")

    try:
        # Build model architecture
        model = build_model(model_type)

        # Load weights if model exists, otherwise use fresh initialization
        if Path(model_path).exists():
            model = load_model_weights(model, model_path, device)
            logger.info(f"  ✓ Loaded model from: {model_path}")
        else:
            logger.info(f"  ⚠️  Using freshly initialized model (no saved weights)")

        # Load public test dataset
        test_loader = get_test_loader(public_test_path)
        logger.info(f"  ✓ Loaded {len(test_loader.dataset)} test samples")

        # Calculate metrics
        metrics = calculate_essential_metrics(model, test_loader, device)

        # Add metadata
        metrics['client_id'] = client_id
        metrics['client_name'] = client_name
        metrics['model_type'] = model_type
        metrics['evaluated_at'] = datetime.now().isoformat()

        logger.info(f"  ✓ Accuracy: {metrics['accuracy']:.2%}, F1: {metrics['f1_score']:.3f}")

        return metrics

    except Exception as e:
        logger.error(f"  ✗ Evaluation failed for {client_name}: {e}")
        raise

def evaluate_all_clients(
    clients: List[Dict],
    public_test_path: str,
    stage: str = "pre_fl"
) -> List[Dict]:
    """
    Evaluate all clients on public test dataset

    Args:
        clients: List of client dictionaries from database
        public_test_path: Path to public test dataset folder
        stage: Either "pre_fl" or "post_fl"

    Returns:
        List of evaluation results for each client
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*70}")
    logger.info(f"{stage.upper()} EVALUATION - Baseline Metrics (Public Test Set)")
    logger.info(f"{'='*70}")
    logger.info(f"Device: {device}")
    logger.info(f"Public Test Path: {public_test_path}")

    # Verify public test path exists
    if not Path(public_test_path).exists():
        raise FileNotFoundError(f"Public test data not found at {public_test_path}")

    results = []

    for client in clients:
        try:
            metrics = evaluate_client(
                client_id=client['id'],
                client_name=client['client_name'],
                model_type=client['model_type'],
                model_path=client['model_path'],
                public_test_path=public_test_path,
                device=device
            )
            results.append(metrics)

        except Exception as e:
            logger.error(f"Failed to evaluate client {client['id']}: {e}")
            # Continue with other clients even if one fails
            continue

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ {stage.upper()} evaluation complete for {len(results)}/{len(clients)} clients")
    logger.info(f"{'='*70}\n")

    return results

def calculate_improvement(pre_metrics: Dict, post_metrics: Dict) -> Dict:
    """Calculate improvement metrics from pre-FL to post-FL"""
    return {
        'accuracy': post_metrics['accuracy'] - pre_metrics['accuracy'],
        'precision': post_metrics['precision'] - pre_metrics['precision'],
        'recall': post_metrics['recall'] - pre_metrics['recall'],
        'f1_score': post_metrics['f1_score'] - pre_metrics['f1_score'],
        'roc_auc': post_metrics['roc_auc'] - pre_metrics['roc_auc'],
        'specificity': post_metrics['specificity'] - pre_metrics['specificity'],
        'healthy_accuracy': post_metrics['healthy_accuracy'] - pre_metrics['healthy_accuracy'],
        'leukemia_accuracy': post_metrics['leukemia_accuracy'] - pre_metrics['leukemia_accuracy'],
    }
