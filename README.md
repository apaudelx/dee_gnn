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

Run from the project root. **All training input flags are required:**

```bash
# Train model (all flags required)
python main.py train \
	--config config/config.json \
	--training-csv data/ee_values_667.csv \
	--nbfix data/NBFIX_table \
	--data-dir data/ee_itp_667
```

You must specify all of:
- `--config` (path to config JSON)
- `--training-csv` (CSV with training data)
- `--nbfix` (NBFIX table file)
- `--data-dir` (directory with compound folders)

If any are missing, the script will show an error and usage instructions.


After training, all outputs (model, config, bead type mapping, metrics, predictions) are saved in a unique subfolder, e.g.:

```
results/20260319_143402/
	model.pth
	config.json
	bead_type_to_id.json
	results.json
	val_predictions.csv
	test_predictions.csv
	val_pred_vs_true.png
	test_pred_vs_true.png
```

## Train/Validation/Test Split Logic

During training, the dataset is split as follows:
- **90%** of compounds are used for training.
- The remaining **10%** are split equally into validation and test sets (**5%** each).
- Splitting is random but reproducible (controlled by the seed).

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

## Bead Type Mapping Consistency (IMPORTANT)

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
python utils/plot_bead_count_vs_encapsulation.py --csv data/ee_values_667.csv --data data
```

The output image (bead_count_vs_encapsulation.png) will be saved in your project root. You can change the output filename with --out.

