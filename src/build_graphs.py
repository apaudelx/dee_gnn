import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
from parse_itp import parse_itp_file


class MolecularGraphBuilder:
    def __init__(self, nbfix_map: Dict[str, Tuple[float, float]], data_dir: str = "training_data",
                 bead_type_to_id: Dict[str, int] = None, extra_data_dirs: List[str] = None):
        self.nbfix_map = nbfix_map
        self.data_dir = data_dir
        self._unknown_beads: Dict[str, List[str]] = {}  # {bead_type: [compound_ids...]}

        if bead_type_to_id is not None:
            self.bead_type_to_id = dict(bead_type_to_id)
            self.num_bead_types = len(self.bead_type_to_id)
            print(f"Loaded bead_type_to_id mapping with {self.num_bead_types} bead types")
        else:
            all_bead_types = set()
            scan_dirs = [data_dir] + (extra_data_dirs or [])
            for d in scan_dirs:
                if not os.path.isdir(d):
                    continue
                for compound_dir in os.listdir(d):
                    itp_path = os.path.join(d, compound_dir, f"{compound_dir}.itp")
                    if os.path.exists(itp_path):
                        data = parse_itp_file(itp_path)
                        for atom in data['atoms']:
                            all_bead_types.add(atom['type'])
            self.bead_type_to_id = {bt: idx for idx, bt in enumerate(sorted(all_bead_types))}
            self.num_bead_types = len(self.bead_type_to_id)
            print(f"Found {self.num_bead_types} unique bead types from {len(scan_dirs)} director{'y' if len(scan_dirs)==1 else 'ies'}")
    
    def get_bead_features(self, bead_type: str, charge: float, mass: float) -> np.ndarray:
        epsilon, sigma = self.nbfix_map.get(bead_type, (0.0, 0.0))
        return np.array([epsilon, sigma, mass, charge])
    
    def build_graph(self, compound_id: str) -> Optional[Data]:
        compound_dir = os.path.join(self.data_dir, compound_id)
        if not os.path.exists(compound_dir):
            return None
        
        itp_files = [f for f in os.listdir(compound_dir) if f.endswith('.itp')]
        
        if len(itp_files) == 0:
            return None
        
        exact_match = f"{compound_id}.itp"
        if exact_match in itp_files:
            itp_path = os.path.join(compound_dir, exact_match)
        else:
            itp_path = os.path.join(compound_dir, itp_files[0])
        
        data = parse_itp_file(itp_path)
        atoms = data['atoms']
        bonds = data['bonds']
        
        if len(atoms) == 0:
            return None
        
        node_features = []
        bead_type_ids = []
        
        for atom in atoms:
            features = self.get_bead_features(atom['type'], atom['charge'], atom['mass'])
            node_features.append(features)
            bead_id = self.bead_type_to_id.get(atom['type'])
            if bead_id is None:
                self._unknown_beads.setdefault(atom['type'], []).append(compound_id)
                bead_id = 0
            bead_type_ids.append(bead_id)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        edge_indices = []
        edge_features = []
        
        for bond in bonds:
            idx_i = bond['ai'] - 1
            idx_j = bond['aj'] - 1
            
            if 0 <= idx_i < len(atoms) and 0 <= idx_j < len(atoms):
                edge_indices.append([idx_i, idx_j])
                edge_indices.append([idx_j, idx_i])
                
                edge_feat = np.array([bond['b0'], bond['k'], float(bond['funct'])], dtype=np.float32)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
        
        if len(edge_indices) == 0:
            edge_indices = np.array([[], []], dtype=np.int64).reshape(2, 0)
            edge_features = np.array([], dtype=np.float32).reshape(0, 3)
        else:
            edge_indices = np.array(edge_indices, dtype=np.int64).T
            edge_features = np.array(edge_features, dtype=np.float32)
        
        degrees = np.zeros(len(atoms), dtype=np.float32)
        if len(edge_indices) > 0:
            for edge in edge_indices.T:
                degrees[edge[0]] += 1.0
        
        node_features = np.column_stack([node_features, degrees])
        
        num_atoms = len(atoms)
        num_bonds = len(bonds)
        avg_degree = np.mean(degrees) if len(degrees) > 0 else 0.0
        max_degree = np.max(degrees) if len(degrees) > 0 else 0.0
        graph_density = (2 * num_bonds) / (num_atoms * (num_atoms - 1)) if num_atoms > 1 else 0.0
        
        charges = [atom['charge'] for atom in atoms]
        total_charge = sum(charges)
        charge_std = np.std(charges) if len(charges) > 0 else 0.0
        
        unique_bead_types = len(set(atom['type'] for atom in atoms))
        
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            bead_type_id=torch.tensor(bead_type_ids, dtype=torch.long),
            edge_index=torch.tensor(edge_indices, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32),
            num_atoms=torch.tensor([num_atoms], dtype=torch.float32),
            num_bonds=torch.tensor([num_bonds], dtype=torch.float32),
            avg_degree=torch.tensor([avg_degree], dtype=torch.float32),
            max_degree=torch.tensor([max_degree], dtype=torch.float32),
            graph_density=torch.tensor([graph_density], dtype=torch.float32),
            total_charge=torch.tensor([total_charge], dtype=torch.float32),
            charge_std=torch.tensor([charge_std], dtype=torch.float32),
            unique_bead_types=torch.tensor([unique_bead_types], dtype=torch.float32),
            compound_id=compound_id
        )
        
        return data
    
    def build_dataset(self, compounds_df: pd.DataFrame) -> List[Data]:
        graphs = []
        missing = []
        
        for _, row in compounds_df.iterrows():
            compound_id = row['compound']
            encapsulation = row['encapsulation_mean']
            
            if pd.isna(encapsulation):
                continue
            
            graph = self.build_graph(compound_id)
            if graph is not None:
                graph.y = torch.tensor([encapsulation], dtype=torch.float32)
                graphs.append(graph)
            else:
                missing.append(compound_id)
        
        if missing:
            print(f"Warning: {len(missing)} compounds not found: {missing[:10]}...")
        
        if self._unknown_beads:
            affected = set()
            for compounds in self._unknown_beads.values():
                affected.update(compounds)

            print(f"\n{'='*60}")
            print(f"FATAL: {len(self._unknown_beads)} bead type(s) not in vocabulary")
            print(f"{'='*60}")
            for bead_type in sorted(self._unknown_beads):
                compounds = sorted(set(self._unknown_beads[bead_type]))
                shown = compounds[:5]
                extra = f" ... (+{len(compounds)-5} more)" if len(compounds) > 5 else ""
                print(f"  {bead_type:10s} found in {len(compounds)} molecule(s): "
                      f"{', '.join(shown)}{extra}")
            print(f"\n  Total affected molecules: {len(affected)}")
            print(f"  Vocabulary has {self.num_bead_types} types, "
                  f"needs {self.num_bead_types + len(self._unknown_beads)}")
            print(f"\n  Fix: retrain with --extra-data-dirs pointing to directories "
                  f"that contain these bead types, or rebuild bead_type_to_id.json")
            print(f"{'='*60}\n")
            raise ValueError(
                f"Unknown bead types: {sorted(self._unknown_beads.keys())}. "
                f"{len(affected)} molecule(s) affected. Cannot proceed with "
                f"corrupted embeddings — see report above."
            )

        print(f"Built {len(graphs)} graphs from {len(compounds_df)} compounds")
        return graphs
