import os
import sys
import argparse
import json
import pandas as pd
import torch
import joblib
import numpy as np
from torch_geometric.loader import DataLoader

from parse_itp import parse_nbfix_table
from build_graphs import MolecularGraphBuilder
from gnn_model import EncapsulationGNN


def load_model(model_path, config_path, device):
    with open(config_path, 'r') as f:
        config = json.load(f)['config']
    model = EncapsulationGNN(
        node_dim=config['node_dim'],
        edge_dim=config['edge_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_bead_types=config['num_bead_types'],
        embedding_dim=config['embedding_dim']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, graphs, device):
    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    preds, ids = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                       batch.bead_type_id, batch.num_atoms, batch.num_bonds,
                       batch.avg_degree, batch.max_degree,
                       batch.graph_density,
                       batch.total_charge, batch.charge_std, batch.unique_bead_types)
            preds.extend(out.cpu().numpy().flatten().tolist())
            ids.extend(batch.compound_id)
    
    return ids, preds


def find_compounds_in_folder(folder_path):
    compounds = []
    if not os.path.exists(folder_path):
        return compounds
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and any(f.endswith('.itp') for f in os.listdir(item_path)):
            compounds.append(item)
    return sorted(compounds)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compounds", "-c", nargs="+", help="Compound IDs")
    parser.add_argument("--file", "-f", help="CSV file with compound column")
    parser.add_argument("--folder", "-d", help="Folder with compound subdirectories")
    parser.add_argument("--use-model", default=None, help="Subfolder in results/ containing model/config/bead_type_to_id.json")
    parser.add_argument("--model", "-m", default=None, help="Model checkpoint path (overrides --use-model)")
    parser.add_argument("--config", default=None, help="Config JSON path (overrides --use-model)")
    parser.add_argument("--nbfix", default="NBFIX_table", help="NBFIX table file path")
    parser.add_argument("--output", "-o", default=None, help="Output CSV (default: predictions.csv in results subdir if set)")
    parser.add_argument("--bead-type-map", default=None, help="Path to bead_type_to_id JSON file from training (overrides --use-model)")

    args = parser.parse_args()

    # If results-subdir is given, set defaults for model/config/bead_type_map/output
    if args.use_model:
        subdir = args.use_model
        if not os.path.isabs(subdir):
            subdir = os.path.join("results", subdir) if not subdir.startswith("results/") else subdir
        model_path = args.model if args.model else os.path.join(subdir, "model.pth")
        config_path = args.config if args.config else os.path.join(subdir, "config.json")
        bead_type_map_path = args.bead_type_map if args.bead_type_map else os.path.join(subdir, "bead_type_to_id.json")
        output_path = args.output if args.output else os.path.join(subdir, "predictions.csv")
    else:
        # Require explicit paths if no results-subdir
        if not args.model or not args.config:
            print("Error: --model and --config are required if --use-model is not provided.")
            sys.exit(1)
        model_path = args.model
        config_path = args.config
        bead_type_map_path = args.bead_type_map
        output_path = args.output if args.output else "predictions.csv"

    if args.folder:
        compound_ids = find_compounds_in_folder(args.folder)
        data_dir = args.folder
    elif args.compounds:
        compound_ids = args.compounds
        data_dir = "training_data"
    elif args.file:
        df = pd.read_csv(args.file)
        if 'compound' not in df.columns:
            print("Error: CSV must have 'compound' column")
            sys.exit(1)
        compound_ids = df['compound'].tolist()
        data_dir = "training_data"
    else:
        print("Error: Provide --folder, --compounds, or --file")
        parser.print_help()
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, config_path, device)

    nbfix_map = parse_nbfix_table(args.nbfix)

    # ── Load scalers saved during training ──────────────────────────────────
    node_scaler  = None
    edge_scaler  = None
    graph_scaler = None

    if args.use_model:
        node_scaler_path  = os.path.join(subdir, "node_scaler.pkl")
        edge_scaler_path  = os.path.join(subdir, "edge_scaler.pkl")
        graph_scaler_path = os.path.join(subdir, "graph_scaler.pkl")

        if os.path.exists(node_scaler_path):
            node_scaler = joblib.load(node_scaler_path)
            print(f"Loaded node scaler from {node_scaler_path}")
        else:
            print("Warning: node_scaler.pkl not found — node features will not be normalized.")

        if os.path.exists(edge_scaler_path):
            edge_scaler = joblib.load(edge_scaler_path)
            print(f"Loaded edge scaler from {edge_scaler_path}")
        else:
            print("Warning: edge_scaler.pkl not found — edge features will not be normalized.")

        if os.path.exists(graph_scaler_path):
            graph_scaler = joblib.load(graph_scaler_path)
            print(f"Loaded graph scaler from {graph_scaler_path}")
        else:
            print("Warning: graph_scaler.pkl not found — graph-level features will not be normalized.")

    bead_type_to_id = None
    if bead_type_map_path:
        if os.path.exists(bead_type_map_path):
            with open(bead_type_map_path, 'r') as f:
                bead_type_to_id = json.load(f)
        else:
            print(f"Warning: bead_type_to_id mapping file not found at {bead_type_map_path}. Proceeding without it.")

    builder = MolecularGraphBuilder(nbfix_map, data_dir=data_dir, bead_type_to_id=bead_type_to_id)

    graphs = []
    for compound_id in compound_ids:
        graph = builder.build_graph(compound_id)
        if graph:
            graphs.append(graph)

    if not graphs:
        print("Error: No graphs built")
        sys.exit(1)

    # ── Apply scalers to inference graphs ───────────────────────────────────
    if node_scaler is not None:
        for g in graphs:
            g.x = torch.tensor(
                node_scaler.transform(g.x.numpy()), dtype=torch.float32)

    if edge_scaler is not None:
        for g in graphs:
            if g.edge_attr.shape[0] > 0:
                g.edge_attr = torch.tensor(
                    edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float32)

    if graph_scaler is not None:
        graph_feats = np.array([[
            g.num_atoms.item(),
            g.num_bonds.item(),
            g.avg_degree.item(),
            g.max_degree.item(),
            g.graph_density.item(),
            g.total_charge.item(),
            g.charge_std.item(),
            g.unique_bead_types.item()
        ] for g in graphs], dtype=np.float32)

        normed = graph_scaler.transform(graph_feats)
        for g, row in zip(graphs, normed):
            g.num_atoms         = torch.tensor([row[0]], dtype=torch.float32)
            g.num_bonds         = torch.tensor([row[1]], dtype=torch.float32)
            g.avg_degree        = torch.tensor([row[2]], dtype=torch.float32)
            g.max_degree        = torch.tensor([row[3]], dtype=torch.float32)
            g.graph_density     = torch.tensor([row[4]], dtype=torch.float32)
            g.total_charge      = torch.tensor([row[5]], dtype=torch.float32)
            g.charge_std        = torch.tensor([row[6]], dtype=torch.float32)
            g.unique_bead_types = torch.tensor([row[7]], dtype=torch.float32)

    compound_ids_pred, predictions = predict(model, graphs, device)

    pd.DataFrame({
        'compound': compound_ids_pred,
        'predicted_encapsulation': predictions
    }).to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
