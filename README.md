# DEE-GNN Quick Run

DEE-GNN is a machine learning framework for predicting molecular encapsulation properties using graph neural networks (GNNs). It builds molecular graphs from input data, trains a GNN model, and enables reproducible inference with organized results. The project is designed for robust, configurable workflows and consistent bead type mapping between training and prediction.

`main.py` is the top-level entrypoint:
- `train`: builds graphs from training data, trains the GNN, and saves all outputs in a unique subfolder under `results/` (e.g., `results/20260319_143402/`).
- `predict`: loads a trained model and writes encapsulation predictions to a CSV file, using outputs from a specific results subfolder.

---




## Setup

It is recommended to use a Python virtual environment for isolation:

```bash
# Create a virtual environment (Python 3.9+ recommended)
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

## Training

Run from the project root. Three data-splitting modes are supported.

### Mode 1: Auto-Split (original)

Pass a single CSV — the script splits it 80/10/10 into train/val/test:

```bash
python main.py train \
	--config config/config.json \
	--training-csv data/ee_values_667.csv \
	--nbfix data/NBFIX_table \
	--data-dir data/ee_itp_667
```

### Mode 2: Custom Split

Provide separate CSVs for training, validation, and/or test sets:

```bash
python main.py train \
	--train-data data/train.csv \
	--val-data data/val.csv \
	--test-data data/test.csv \
	--nbfix data/NBFIX_table \
	--data-dir data/ee_itp_667
```

You may omit `--test-data` (no test evaluation) or `--val-data` (see train-only below).

### Mode 3: Train-Only (no splitting)

Use 100% of the data for training. Because there is no validation set for early stopping, `--epochs` is required:

```bash
python main.py train \
	--train-data data/ee_values_667.csv \
	--epochs 500 \
	--nbfix data/NBFIX_table \
	--data-dir data/ee_itp_667
```

### Common Flags

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes (auto-supplied by `main.py`) | Path to config JSON |
| `--nbfix` | Yes (auto-supplied by `main.py`) | NBFIX table file |
| `--data-dir` | Yes | Directory with compound folders |
| `--training-csv` | Mode 1 only | Single CSV — auto-split 80/10/10 |
| `--train-data` | Modes 2 & 3 | CSV for training set |
| `--val-data` | Optional (Mode 2) | CSV for validation set |
| `--test-data` | Optional (Mode 2) | CSV for test set |
| `--epochs` | Required for Mode 3 | Fixed epoch count (overrides config `max_epochs`) |
| `--results-dir` | No (default: `results`) | Output directory |
| `--seed` | No (default: 121) | Random seed |

> `--training-csv` and `--train-data` are mutually exclusive.

After training, all outputs (model, config, bead type mapping, metrics, predictions) are saved in a unique subfolder, e.g.:

```
results/20260319_143402/
	model.pth
	config.json
	bead_type_to_id.json
	results.json
	val_predictions.csv   # only when val data is present
	test_predictions.csv  # only when test data is present
	val_pred_vs_true.png  # only when val data is present
	test_pred_vs_true.png # only when test data is present
```

## Train/Validation/Test Split Logic

When using `--training-csv` (auto-split mode), the dataset is split as follows:
- **90%** of compounds are used for training.
- The remaining **10%** are split equally into validation and test sets (**5%** each).
- Splitting is random but reproducible (controlled by the seed).

When using `--train-data`, you control the splits entirely via the CSVs you provide.

## Inference / Prediction

You can now use the `--use-model` flag to specify the subfolder from a previous run. This will automatically use the correct model, config, and bead type mapping for inference.

```bash
# Predict from a folder of compound subdirectories
python main.py predict --use-model results/20260319_143402 --folder training_data

# Predict from explicit compound IDs
python main.py predict --use-model results/20260319_143402 --compounds HEaSC00031 HEaSC00033

# Predict from a CSV with a 'compound' column
python main.py predict --use-model results/20260319_143402 --file test_predictions.csv
```

By default, predictions will be saved as `predictions.csv` in the same subfolder. You can override the output path with `--output` if desired.

### Advanced: Overriding Defaults

You can still override any file (model, config, bead type map) by passing `--model`, `--config`, or `--bead-type-map` explicitly. If you do not use `--use-model`, you must provide all required files manually.

---

## Bead Type Mapping Consistency

The bead type ID mapping used during training is saved as `bead_type_to_id.json` in the results subfolder. Using `--use-model` ensures this mapping is used for inference, guaranteeing bead type IDs match between training and prediction, and preventing embedding mismatches.


# Utility

## Bead Count vs Encapsulation Plot 
This utility script visualizes the relationship between bead count and encapsulation for your dataset.

- It reads a CSV file with compound IDs and encapsulation values.
- It scans .itp files in your data folders to count beads for each compound.
- It produces a PNG plot with bead count distribution, encapsulation distribution, and a scatter plot of bead count vs encapsulation.

**How to run:**

```bash
python utils/plot_bead_count_vs_encapsulation.py --csv <your_csv> --data <your_data_folder>
```

Example:

```bash
python utils/plot_bead_count_vs_encapsulation.py --csv data/ee_values_667.csv --data data/ee_itp_667/
```

By default, the output image (bead_count_vs_encapsulation.png) will be saved in the root dir. You can change the output filename and location with --out.

