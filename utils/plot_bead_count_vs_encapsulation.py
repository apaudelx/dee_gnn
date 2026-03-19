#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

SECTION_HEADER = "["
COMMENT_PREFIXES = (";", "#")


def iter_atoms_section(lines):
    in_atoms = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith(SECTION_HEADER) and s.endswith("]"):
            in_atoms = s.strip("[] ").lower() == "atoms"
            continue
        if not in_atoms:
            continue
        if s.startswith(COMMENT_PREFIXES):
            continue
        yield s


def count_beads_in_itp(itp_path: Path) -> int:
    lines = itp_path.read_text(encoding="utf-8", errors="replace").splitlines()
    count = 0
    for s in iter_atoms_section(lines):
        parts = s.split()
        if len(parts) >= 2:
            count += 1
    return count


def find_itp(compound_dir: Path) -> Optional[Path]:
    exact = compound_dir / f"{compound_dir.name}.itp"
    if exact.is_file():
        return exact
    for itp in compound_dir.glob("*.itp"):
        return itp
    return None


def build_bead_counts(data_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    subdirs = [data_root / "2D_molecules", data_root / "Working_5_to_10_pesticides"]
    for subdir in subdirs:
        if not subdir.is_dir():
            continue
        for compound_dir in subdir.iterdir():
            if not compound_dir.is_dir():
                continue
            itp_path = find_itp(compound_dir)
            if itp_path is None:
                continue
            counts[compound_dir.name] = count_beads_in_itp(itp_path)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with compound and encapsulation_mean")
    ap.add_argument("--data", default="filtered_training_data", help="Data root with .itp files")
    ap.add_argument("--out", default="bead_count_vs_encapsulation.png", help="Output image")
    ap.add_argument("--id-col", default="compound", help="Compound id column")
    ap.add_argument("--y-col", default="encapsulation_mean", help="Target column")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.id_col not in df.columns or args.y_col not in df.columns:
        raise SystemExit("CSV must contain id and target columns")

    counts = build_bead_counts(Path(args.data))
    df["bead_count"] = df[args.id_col].map(counts)
    df = df.dropna(subset=["bead_count", args.y_col])

    if df.empty:
        raise SystemExit("No rows matched between CSV and data folder")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(df["bead_count"], bins=9, color="#1f77b4", alpha=0.85)
    plt.xlabel("Bead count")
    plt.ylabel("Molecules")
    plt.title("Bead Count Distribution")

    plt.subplot(1, 3, 2)
    plt.hist(df[args.y_col], bins=20, color="#ff7f0e", alpha=0.85)
    plt.xlabel(args.y_col)
    plt.ylabel("Molecules")
    plt.title("Encapsulation Distribution")

    plt.subplot(1, 3, 3)
    plt.scatter(df["bead_count"], df[args.y_col], s=12, alpha=0.7, color="#2ca02c")
    plt.xlabel("Bead count")
    plt.ylabel(args.y_col)
    plt.title("Bead Count vs Encapsulation")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote plot to {args.out}")


if __name__ == "__main__":
    main()
