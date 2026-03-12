"""
Usage: python3 info_1D_dataset.py [PATH]

Show basic info about a 1D time-series dataset file.
Supports numpy (.npy/.npz), pickle (.pkl/.pickle), and h5 (.h5/.hdf5).

Example:
python3 info_1D_dataset.py data/proprioceptive_data/combined.pkl
python3 info_1D_dataset.py data/proprioceptive_data/combined.npz
python3 info_1D_dataset.py data/proprioceptive_data/combined.h5
"""
import argparse
from pathlib import Path
from typing import Dict, Literal

import numpy as np

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

try:
    import pickle
except ImportError:  # pragma: no cover - extremely unlikely
    pickle = None


FileType = Literal["numpy", "h5", "pickle"]


def detect_file_type(path: Path) -> FileType:
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return "numpy"
    if suffix in {".h5", ".hdf5"}:
        return "h5"
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    # default: try numpy, then pickle
    return "numpy"


def load_series_dict(path: Path) -> Dict[str, np.ndarray]:
    """
    Load a file that contains a dict of 1D time series, or a single 1D array.

    Returns: mapping from series name to 1D numpy array.
    """
    ftype = detect_file_type(path)
    series: Dict[str, np.ndarray] = {}

    def to_array(value: object) -> np.ndarray | None:
        # Handle list/tuple of arrays (e.g. sequence of segments)
        if isinstance(value, (list, tuple)):
            parts = []
            for v in value:
                a = np.asarray(v)
                if a.ndim == 0:
                    a = a.reshape(1)
                parts.append(a)
            if not parts:
                return None
            return np.concatenate(parts, axis=0)

        # Handle object arrays that are sequences
        if isinstance(value, np.ndarray) and value.dtype == object:
            parts = []
            for v in value:
                a = np.asarray(v)
                if a.ndim == 0:
                    a = a.reshape(1)
                parts.append(a)
            if not parts:
                return None
            return np.concatenate(parts, axis=0)

        a = np.asarray(value)
        return a

    if ftype == "numpy":
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=True)
            for key in data.files:
                a = to_array(data[key])
                if a is not None:
                    series[key] = a
        else:  # .npy or unknown
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, dict):
                for key, value in arr.items():
                    a = to_array(value)
                    if a is not None:
                        series[str(key)] = a
            else:
                a = to_array(arr)
                if a is not None:
                    series["data"] = a
    elif ftype == "pickle":
        if pickle is None:
            raise RuntimeError("pickle module not available.")
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for key, value in obj.items():
                a = to_array(value)
                if a is not None:
                    series[str(key)] = a
        else:
            a = to_array(obj)
            if a is not None:
                series["data"] = a
    else:  # "h5"
        if h5py is None:
            raise RuntimeError("h5py is not installed. Install with `pip install h5py`.")
        with h5py.File(path, "r") as f:
            for name, obj in f.items():
                if isinstance(obj, h5py.Dataset):
                    a = np.asarray(obj[()])
                    if a.ndim == 1:
                        series[name] = a
 
    if not series:
        raise ValueError(f"No 1D time series found in file {path}")

    return series


def print_info(path: Path) -> None:
    series = load_series_dict(path)

    print(f"File: {path}")
    print(f"Number of 1D series: {len(series)}")
    print()
    print(f"{'Key':30s} {'Length':>10s} {'Dtype':>10s}")
    print("-" * 54)

    total_length = 0
    for key, arr in sorted(series.items()):
        length = len(arr)
        total_length += length
        print(f"{key:30s} {length:10d} {str(arr.dtype):>10s}")

    print("-" * 54)
    print(f"{'TOTAL':30s} {total_length:10d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show basic info about a 1D time-series dataset file.\n"
            "Supports numpy (.npy/.npz), pickle (.pkl/.pickle), and h5 (.h5/.hdf5).\n"
            "Prints all series keys and their lengths."
        )
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the dataset file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.file).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"File does not exist: {path}")

    print_info(path)


if __name__ == "__main__":
    main()

