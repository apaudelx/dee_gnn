import re
from typing import Dict, Tuple


def parse_itp_file(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        content = f.read()
    
    atoms = []
    atoms_section = re.search(r'\[ atoms \](.*?)(?=\[|\Z)', content, re.DOTALL)
    if atoms_section:
        for line in atoms_section.group(1).strip().split('\n'):
            line = line.strip()
            if line and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 8:
                    atoms.append({
                        'nr': int(parts[0]),
                        'type': parts[1],
                        'charge': float(parts[6]),
                        'mass': float(parts[7])
                    })
    
    bonds = []
    bonds_section = re.search(r'\[ bonds \](.*?)(?=\[|\Z)', content, re.DOTALL)
    if bonds_section:
        for line in bonds_section.group(1).strip().split('\n'):
            line = line.strip()
            if line and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 5:
                    bonds.append({
                        'ai': int(parts[0]),
                        'aj': int(parts[1]),
                        'funct': int(parts[2]),
                        'b0': float(parts[3]),
                        'k': float(parts[4])
                    })
    
    return {'atoms': atoms, 'bonds': bonds}


def parse_nbfix_table(filepath: str) -> Dict[str, Tuple[float, float]]:
    nbfix_map = {}
    
    with open(filepath, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            if not line or line.startswith('NBFIX'):
                continue
            
            parts = line.split()
            if len(parts) >= 4 and parts[0] == parts[1]:
                try:
                    nbfix_map[parts[0]] = (float(parts[2]), float(parts[3]))
                except (ValueError, IndexError):
                    continue
    
    return nbfix_map
