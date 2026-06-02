import torch
from torch_geometric.utils import to_undirected


def load_and_preprocess_data(
    data_path: str = '../data/processed/pyg_data.pt',
    device: torch.device | None = None,
) -> tuple:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data = torch.load(data_path)
    data.edge_index = to_undirected(data.edge_index)

    train_features = data.x[data.train_mask]
    train_mean = train_features.mean(dim=0)
    train_std = train_features.std(dim=0)
    data.x = (data.x - train_mean) / (train_std + 1e-8)
    print(data)

    data = data.to(device)

    return data, device
