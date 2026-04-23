import os
import random
import argparse
import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.stats import spearmanr
import json
from datetime import datetime
from pathlib import Path

from parse_itp import parse_nbfix_table
from build_graphs import MolecularGraphBuilder
from gnn_model import EncapsulationGNN


DEFAULT_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'max_epochs': 1000,
    'early_stopping_patience': 100,
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.2,
    'node_dim': 5,
    'num_bead_types': 100,
    'embedding_dim': 32,
    'edge_dim': 3
}


def load_config(config_path=None):
    config = dict(DEFAULT_CONFIG)
    if not config_path:
        config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.json'

    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).resolve().parent.parent / config_file

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if isinstance(raw, dict) and 'config' in raw and isinstance(raw['config'], dict):
        loaded = raw['config']
    elif isinstance(raw, dict):
        loaded = raw
    else:
        raise ValueError("Invalid config format. Expected JSON object.")

    config.update(loaded)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    try:
        spearman_corr, _ = spearmanr(y_true, y_pred)
        spearman_corr = 0.0 if np.isnan(spearman_corr) else spearman_corr
    except:
        spearman_corr = 0.0
    
    return {'mae': float(mae), 'mse': float(mse), 'r2': float(r2), 'spearman': float(spearman_corr)}


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                   batch.bead_type_id, batch.num_atoms, batch.num_bonds,
                   batch.avg_degree, batch.max_degree, batch.graph_density,
                   batch.total_charge, batch.charge_std, batch.unique_bead_types)
        
        loss = criterion(out.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device):
    """Return (metrics, preds, targets, compound_ids).

    compound_ids is aligned with preds/targets when each batch exposes
    ``compound_id`` (list of str, same length as preds); otherwise None.
    """
    model.eval()
    all_preds, all_targets, all_ids = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                       batch.bead_type_id, batch.num_atoms, batch.num_bonds,
                       batch.avg_degree, batch.max_degree, batch.graph_density,
                       batch.total_charge, batch.charge_std, batch.unique_bead_types)
            all_preds.append(out.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            cid = getattr(batch, "compound_id", None)
            if cid is not None:
                if isinstance(cid, str):
                    all_ids.append(cid)
                else:
                    all_ids.extend(list(cid))

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    compound_ids = all_ids if len(all_ids) == len(all_preds) else None
    return compute_metrics(all_targets, all_preds), all_preds, all_targets, compound_ids


def train_model(model, train_loader, val_loader, device, config):
    def weighted_mse_loss(pred, target):
        base_weights = 1.0 + 2.5 * target
        error_magnitude = torch.abs(pred - target)
        error_penalty = 1.0 + 0.5 * torch.clamp(error_magnitude - 0.2, min=0.0)
        weights = base_weights * error_penalty
        return torch.mean(weights * (pred - target) ** 2)
    
    criterion = weighted_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_only = val_loader is None
    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses = []
    
    for epoch in range(config['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        if train_only:
            scheduler.step(train_loss)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{config['max_epochs']}: Train Loss={train_loss:.4f}")
            continue

        val_metrics, _, _, _ = validate(model, val_loader, device)
        scheduler.step(val_metrics['mae'])
        
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            if epoch >= 50:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                patience_counter = 0
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val MAE={val_metrics['mae']:.4f}, "
                  f"Val R²={val_metrics['r2']:.4f}, Val Spearman={val_metrics['spearman']:.4f}")
    
    if train_only:
        return None, None, None, None, train_losses

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    metrics, preds, targets, val_compounds = validate(model, val_loader, device)
    return metrics, preds, targets, val_compounds, train_losses


def _fit_scaler(features, label):
    """Fit a StandardScaler and patch zero-variance columns."""
    scaler = StandardScaler()
    scaler.fit(features)
    zero_var = np.where(scaler.scale_ == 0)[0]
    if len(zero_var) > 0:
        print(f"  Warning: zero-variance {label} feature columns (indices): {zero_var.tolist()}")
        scaler.scale_[zero_var] = 1.0
        scaler.mean_[zero_var]  = 0.0
    return scaler


def get_graph_level_feats(graphs):
    return np.array([[
        g.num_atoms.item(),
        g.num_bonds.item(),
        g.avg_degree.item(),
        g.max_degree.item(),
        g.graph_density.item(),
        g.total_charge.item(),
        g.charge_std.item(),
        g.unique_bead_types.item()
    ] for g in graphs], dtype=np.float32)


def apply_graph_scaler(graphs, scaler):
    feats  = get_graph_level_feats(graphs)
    normed = scaler.transform(feats)
    for g, row in zip(graphs, normed):
        g.num_atoms         = torch.tensor([row[0]], dtype=torch.float32)
        g.num_bonds         = torch.tensor([row[1]], dtype=torch.float32)
        g.avg_degree        = torch.tensor([row[2]], dtype=torch.float32)
        g.max_degree        = torch.tensor([row[3]], dtype=torch.float32)
        g.graph_density     = torch.tensor([row[4]], dtype=torch.float32)
        g.total_charge      = torch.tensor([row[5]], dtype=torch.float32)
        g.charge_std        = torch.tensor([row[6]], dtype=torch.float32)
        g.unique_bead_types = torch.tensor([row[7]], dtype=torch.float32)


def normalize_features(train_graphs, extra_graphs_lists):
    """Fit scalers on train_graphs, then apply to train + all extra lists."""
    all_graphs = list(train_graphs)
    for gl in extra_graphs_lists:
        all_graphs.extend(gl)

    node_scaler = _fit_scaler(
        np.vstack([g.x.numpy() for g in train_graphs]), "node")
    for g in all_graphs:
        g.x = torch.tensor(
            node_scaler.transform(g.x.numpy()), dtype=torch.float32)
    print(f"  Node features normalized (fit on {len(train_graphs)} training graphs)")

    edge_scaler = _fit_scaler(
        np.vstack([g.edge_attr.numpy() for g in train_graphs
                   if g.edge_attr.shape[0] > 0]), "edge")
    for g in all_graphs:
        if g.edge_attr.shape[0] > 0:
            g.edge_attr = torch.tensor(
                edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float32)
    print(f"  Edge features normalized (fit on edges from {len(train_graphs)} training graphs)")

    graph_scaler = _fit_scaler(get_graph_level_feats(train_graphs), "graph-level")
    for gl in [train_graphs] + extra_graphs_lists:
        apply_graph_scaler(gl, graph_scaler)
    print(f"  Graph-level features normalized (fit on {len(train_graphs)} training graphs)")

    return node_scaler, edge_scaler, graph_scaler


def _plot_pred_vs_true(targets, preds, metrics, title, color, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, s=12, alpha=0.7, color=color)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--', lw=2)
    plt.xlabel("True encapsulation")
    plt.ylabel("Predicted encapsulation")
    plt.title(title)
    rmse = np.sqrt(metrics['mse'])
    plt.legend([f"RMSE: {rmse:.4f}\nMAE: {metrics['mae']:.4f}"], loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_training_loss(train_losses, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, color="#2ca02c", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train encapsulation prediction GNN")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--nbfix", required=True, help="Path to NBFIX table")
    parser.add_argument("--data-dir", required=True, help="Directory containing compound folders")
    parser.add_argument("--extra-data-dirs", nargs="*", default=[],
                        help="Additional data directories (bead types from all dirs are included in vocabulary)")
    parser.add_argument("--results-dir", default="results", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=121, help="Random seed")

    split_group = parser.add_argument_group("data split options (choose one mode)")
    split_group.add_argument("--training-csv", default=None,
                             help="Single CSV with all compounds — random auto-split 80/10/10")
    split_group.add_argument("--train-data", default=None,
                             help="CSV for training set (use alone for train-only, or with --val-data/--test-data)")
    split_group.add_argument("--val-data", default=None,
                             help="CSV for validation set (requires --train-data)")
    split_group.add_argument("--test-data", default=None,
                             help="CSV for test set (requires --train-data)")
    split_group.add_argument("--epochs", type=int, default=None,
                             help="Fixed number of training epochs (overrides config max_epochs; "
                                  "required for train-only mode when no val set is provided)")

    args = parser.parse_args()

    # ── Validate argument combinations ───────────────────────────────────────
    if args.training_csv and args.train_data:
        parser.error("--training-csv and --train-data are mutually exclusive.")
    if not args.training_csv and not args.train_data:
        parser.error("Provide either --training-csv (auto-split) or --train-data (custom/train-only).")
    if (args.val_data or args.test_data) and not args.train_data:
        parser.error("--val-data/--test-data require --train-data.")
    train_only = args.train_data and not args.val_data and not args.test_data
    if train_only and args.epochs is None:
        parser.error("--epochs is required for train-only mode (no val/test data for early stopping).")

    config = load_config(args.config)
    if args.epochs is not None:
        config['max_epochs'] = args.epochs

    print("Encapsulation Prediction GNN Training")

    seed = args.seed
    set_seed(seed)

    nbfix_map = parse_nbfix_table(args.nbfix)
    print(f"Loaded {len(nbfix_map)} bead type parameters")

    builder = MolecularGraphBuilder(nbfix_map, data_dir=args.data_dir,
                                     extra_data_dirs=args.extra_data_dirs)

    # ── Build graph sets depending on split mode ─────────────────────────────
    if args.training_csv:
        # Auto-split mode: single CSV -> 80/10/10 (random by graph index)
        compounds_df = pd.read_csv(args.training_csv)
        compounds_df = compounds_df.dropna(subset=['encapsulation_mean'])
        print(f"\nLoaded {len(compounds_df)} compounds from {args.training_csv}")

        graphs = builder.build_dataset(compounds_df)
        if len(graphs) == 0:
            print("Error: No graphs were built!")
            return

        train_idx, temp_idx = train_test_split(
            range(len(graphs)), test_size=0.2, random_state=seed, shuffle=True)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, shuffle=True)

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs   = [graphs[i] for i in val_idx]
        test_graphs  = [graphs[i] for i in test_idx]

    else:
        # Custom / train-only mode
        train_df = pd.read_csv(args.train_data).dropna(subset=['encapsulation_mean'])
        print(f"\nLoaded {len(train_df)} training compounds from {args.train_data}")
        train_graphs = builder.build_dataset(train_df)

        val_graphs  = []
        test_graphs = []

        if args.val_data:
            val_df = pd.read_csv(args.val_data).dropna(subset=['encapsulation_mean'])
            print(f"Loaded {len(val_df)} validation compounds from {args.val_data}")
            val_graphs = builder.build_dataset(val_df)

        if args.test_data:
            test_df = pd.read_csv(args.test_data).dropna(subset=['encapsulation_mean'])
            print(f"Loaded {len(test_df)} test compounds from {args.test_data}")
            test_graphs = builder.build_dataset(test_df)

    if len(train_graphs) == 0:
        print("Error: No training graphs were built!")
        return

    config['num_bead_types'] = builder.num_bead_types

    if train_only:
        print(f"\nTrain-only mode: {len(train_graphs)} graphs, {config['max_epochs']} epochs")
    else:
        print(f"\nTrain: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # ── Feature normalization ────────────────────────────────────────────────
    extra = [g for g in [val_graphs, test_graphs] if g]
    node_scaler, edge_scaler, graph_scaler = normalize_features(train_graphs, extra)

    # ── Data loaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(train_graphs, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_graphs,  batch_size=config['batch_size'], shuffle=False) if val_graphs else None
    test_loader  = DataLoader(test_graphs, batch_size=config['batch_size'], shuffle=False) if test_graphs else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print("\nTraining...")

    model = EncapsulationGNN(
        node_dim=config['node_dim'],
        edge_dim=config['edge_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_bead_types=config['num_bead_types'],
        embedding_dim=config['embedding_dim']
    ).to(device)

    val_metrics, val_preds, val_targets, val_compounds, train_losses = train_model(
        model, train_loader, val_loader, device, config)

    # ── Evaluate on training data (in-sample; shuffle=False preserves graph / compound order)
    eval_train_loader = DataLoader(train_graphs, batch_size=config['batch_size'], shuffle=False)
    train_metrics, train_preds, train_targets, train_compounds = validate(
        model, eval_train_loader, device)

    # ── Evaluate test set if present ─────────────────────────────────────────
    test_metrics, test_preds, test_targets, test_compounds = (None, None, None, None)
    if test_loader is not None:
        test_metrics, test_preds, test_targets, test_compounds = validate(
            model, test_loader, device)

    # ── Print results ────────────────────────────────────────────────────────
    print("RESULTS")
    print("\nTraining Set:")
    print(f"  MAE: {train_metrics['mae']:.4f}, RMSE: {np.sqrt(train_metrics['mse']):.4f}")
    print(f"  R²: {train_metrics['r2']:.4f}, Spearman: {train_metrics['spearman']:.4f}")
    if val_metrics:
        print("\nValidation Set:")
        print(f"  MAE: {val_metrics['mae']:.4f}, RMSE: {np.sqrt(val_metrics['mse']):.4f}")
        print(f"  R²: {val_metrics['r2']:.4f}, Spearman: {val_metrics['spearman']:.4f}")
    if test_metrics:
        print("\nTest Set:")
        print(f"  MAE: {test_metrics['mae']:.4f}, RMSE: {np.sqrt(test_metrics['mse']):.4f}")
        print(f"  R²: {test_metrics['r2']:.4f}, Spearman: {test_metrics['spearman']:.4f}")

    # ── Save artifacts ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_subdir = os.path.join(args.results_dir, timestamp)
    os.makedirs(run_subdir, exist_ok=True)

    bead_type_map_path = os.path.join(run_subdir, "bead_type_to_id.json")
    with open(bead_type_map_path, 'w') as f:
        json.dump(builder.bead_type_to_id, f, indent=2)

    with open(os.path.join(run_subdir, "config.json"), 'w') as f:
        json.dump({'config': config}, f, indent=2)

    results_payload = {'config': config, 'train_metrics': train_metrics}
    if val_metrics:
        results_payload['val_metrics'] = val_metrics
    if test_metrics:
        results_payload['test_metrics'] = test_metrics
    with open(os.path.join(run_subdir, "results.json"), 'w') as f:
        json.dump(results_payload, f, indent=2)

    if train_preds is not None:
        train_df = {'target': train_targets, 'predicted': train_preds}
        if train_compounds is not None:
            train_df['compound'] = train_compounds
        pd.DataFrame(train_df).to_csv(
            os.path.join(run_subdir, "train_predictions.csv"), index=False)
        train_plot_path = os.path.join(run_subdir, "train_pred_vs_true.png")
        _plot_pred_vs_true(train_targets, train_preds, train_metrics,
                           "Training: Predicted vs True", "#2ca02c", train_plot_path)

    if train_losses:
        loss_plot_path = os.path.join(run_subdir, "training_loss.png")
        _plot_training_loss(train_losses, loss_plot_path)

    if val_preds is not None:
        val_df = {'target': val_targets, 'predicted': val_preds}
        if val_compounds is not None:
            val_df['compound'] = val_compounds
        pd.DataFrame(val_df).to_csv(
            os.path.join(run_subdir, "val_predictions.csv"), index=False)
        val_plot_path = os.path.join(run_subdir, "val_pred_vs_true.png")
        _plot_pred_vs_true(val_targets, val_preds, val_metrics,
                           "Validation: Predicted vs True", "#1f77b4", val_plot_path)

    if test_preds is not None:
        test_df = {'target': test_targets, 'predicted': test_preds}
        if test_compounds is not None:
            test_df['compound'] = test_compounds
        pd.DataFrame(test_df).to_csv(
            os.path.join(run_subdir, "test_predictions.csv"), index=False)
        test_plot_path = os.path.join(run_subdir, "test_pred_vs_true.png")
        _plot_pred_vs_true(test_targets, test_preds, test_metrics,
                           "Test: Predicted vs True", "#ff7f0e", test_plot_path)

    torch.save(model.state_dict(), os.path.join(run_subdir, "model.pth"))

    joblib.dump(node_scaler,  os.path.join(run_subdir, "node_scaler.pkl"))
    joblib.dump(edge_scaler,  os.path.join(run_subdir, "edge_scaler.pkl"))
    joblib.dump(graph_scaler, os.path.join(run_subdir, "graph_scaler.pkl"))

    print(f"\nAll results saved in: {run_subdir}")
    print(f"  - Model: {os.path.join(run_subdir, 'model.pth')}")
    print(f"  - Config: {os.path.join(run_subdir, 'config.json')}")
    print(f"  - Bead type mapping: {bead_type_map_path}")
    print(f"  - Metrics: {os.path.join(run_subdir, 'results.json')}")
    print(f"  - Node scaler:  {os.path.join(run_subdir, 'node_scaler.pkl')}")
    print(f"  - Edge scaler:  {os.path.join(run_subdir, 'edge_scaler.pkl')}")
    print(f"  - Graph scaler: {os.path.join(run_subdir, 'graph_scaler.pkl')}")
    if train_losses:
        print(f"  - Training loss curve: {loss_plot_path}")
    if train_preds is not None:
        print(f"  - Training predictions: {os.path.join(run_subdir, 'train_predictions.csv')}")
        print(f"  - Training plot: {train_plot_path}")
    if val_preds is not None:
        print(f"  - Validation predictions: {os.path.join(run_subdir, 'val_predictions.csv')}")
        print(f"  - Validation plot: {val_plot_path}")
    if test_preds is not None:
        print(f"  - Test predictions: {os.path.join(run_subdir, 'test_predictions.csv')}")
        print(f"  - Test plot: {test_plot_path}")


if __name__ == "__main__":
    main()
