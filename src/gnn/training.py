import math
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix, classification_report
)


def compute_class_weights(data, device: torch.device) -> torch.Tensor:
    n_licit = (data.y[data.train_mask] == 0).sum().item()
    n_illicit = (data.y[data.train_mask] == 1).sum().item()
    weight = torch.tensor(
        [1.0, math.sqrt(n_licit / n_illicit)],
        dtype=torch.float
    ).to(device)

    return weight


def evaluate(model, data, criterion, mask) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        probs = F.softmax(out[mask], dim=1)[:, 1]
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = probs.cpu().numpy()
        loss = criterion(out[mask], data.y[mask]).item()

    acc = accuracy_score(y_true, y_pred)
    precision_ill = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_ill = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_ill = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc_pr = average_precision_score(y_true, y_prob)

    return {
        'loss': loss, 'accuracy': acc,
        'precision_ill': precision_ill, 'recall_ill': recall_ill,
        'f1_ill': f1_ill, 'f1_macro': f1_macro, 'auc_pr': auc_pr,
    }


def train_one_epoch(model, data, optimizer, criterion, clip_grad_norm=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        
    optimizer.step()
    return loss.item()


def train_with_early_stopping(
    model,
    data,
    optimizer,
    criterion,
    save_path: str,
    num_epochs: int = 200,
    patience: int = 20,
    monitor_metric: str = 'f1_ill',
    clip_grad_norm=None,
) -> dict:
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    best_val_score = 0.0
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, data, optimizer, criterion, clip_grad_norm=clip_grad_norm)

        train_metrics = evaluate(model, data, criterion, data.train_mask)
        val_metrics = evaluate(model, data, criterion, data.val_mask)

        train_f1 = train_metrics['f1_ill']
        val_f1 = val_metrics['f1_ill']
        val_loss = val_metrics['loss']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        if val_metrics[monitor_metric] > best_val_score:
            best_val_score = val_metrics[monitor_metric]
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:03d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
              f"Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, "
              f"Gap={train_f1 - val_f1:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (Best epoch: {best_epoch})")
            break

    return history

def print_test_evaluation(model, data, criterion, label: str = "Test") -> None:
    test_metrics = evaluate(model, data, criterion, data.test_mask)

    print(f"{label}")
    print(f"  Loss          : {test_metrics['loss']:.4f}")
    print(f"  Accuracy      : {test_metrics['accuracy']:.4f}")
    print(f"  Precision(ill): {test_metrics['precision_ill']:.4f}")
    print(f"  Recall(ill)   : {test_metrics['recall_ill']:.4f}")
    print(f"  F1(illicit)   : {test_metrics['f1_ill']:.4f}")
    print(f"  F1(macro)     : {test_metrics['f1_macro']:.4f}")
    print(f"  AUC-PR        : {test_metrics['auc_pr']:.4f}")

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[data.test_mask].argmax(dim=1)
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred.cpu().numpy()

    print("Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Licit (0)', 'Illicit (1)'],
        digits=4,
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
