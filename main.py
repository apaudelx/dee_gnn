import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.json"


def run_script(script_name: str, extra_args: list) -> None:
	script_path = SRC_DIR / script_name
	if not script_path.exists():
		raise FileNotFoundError(f"Script not found: {script_path}")

	cmd = [sys.executable, str(script_path), *extra_args]
	subprocess.run(cmd, check=True)


def ensure_predict_defaults(args: list) -> list:
	has_config = any(a == "--config" or a.startswith("--config=") for a in args)
	has_nbfix = any(a == "--nbfix" or a.startswith("--nbfix=") for a in args)
	has_use_model = any(a == "--use-model" or a.startswith("--use-model=") for a in args)

	final_args = list(args)
	# Only add --config if not present and --use-model is not present
	if not has_config and not has_use_model:
		final_args.extend(["--config", str(DEFAULT_CONFIG)])
	if not has_nbfix:
		final_args.extend(["--nbfix", "data/NBFIX_table"])

	return final_args


def ensure_train_defaults(args: list) -> list:
	has_config = any(a == "--config" or a.startswith("--config=") for a in args)
	has_nbfix = any(a == "--nbfix" or a.startswith("--nbfix=") for a in args)

	final_args = list(args)
	if not has_config:
		final_args.extend(["--config", str(DEFAULT_CONFIG)])
	if not has_nbfix:
		final_args.extend(["--nbfix", "data/NBFIX_table"])

	return final_args


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Abstract project entrypoint. Delegates to src/train.py and src/inference.py"
	)
	parser.add_argument("task", choices=["train", "predict"], help="Task to run")
	parsed, extra = parser.parse_known_args()

	if parsed.task == "train":
		run_script("train.py", ensure_train_defaults(extra))
		return

	run_script("inference.py", ensure_predict_defaults(extra))


if __name__ == "__main__":
	main()
