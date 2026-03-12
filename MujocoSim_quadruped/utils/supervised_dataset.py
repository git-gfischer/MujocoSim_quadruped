import os
import gzip
import pickle as pkl
import datetime
import numbers
from dataclasses import dataclass, asdict
from collections import deque
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


# ============================================================
# Field definition
# ============================================================

@dataclass
class DataField:
    """
    Describes one feature or label to be extracted and stored.

    Attributes
    ----------
    name : str
        Name used in the saved dataset.
    source : str
        Slash-separated path inside nested dictionaries.
        Example: "imu/angular_velocity/z"
    kind : str
        One of:
            - "feature"
            - "label"
    extract : str
        How to extract from source:
            - "last"   : take the last element if deque/list/array
            - "all"    : take full content
            - "value"  : take as-is
    dtype : Any
        Desired numpy dtype (e.g. np.float32, np.int8).
    shape : Optional[tuple]
        Optional expected shape. If provided, data is validated.
    required : bool
        If True, missing field raises an error.
        If False, missing field is skipped or filled with None.
    """
    name: str
    source: str
    kind: str = "feature"
    extract: str = "last"
    dtype: Any = np.float32
    shape: Optional[tuple] = None
    required: bool = True


# ============================================================
# Utility helpers
# ============================================================

def _now_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _resolve_nested_path(data: Dict[str, Any], path: str) -> Any:
    """
    Resolve slash-separated path inside nested dicts.

    Example:
        path = "imu/angular_velocity/z"
    """
    node = data
    for key in path.split("/"):
        if not isinstance(node, dict):
            raise KeyError(f"Path '{path}' failed at '{key}': node is not a dict.")
        if key not in node:
            raise KeyError(f"Missing key '{key}' in path '{path}'.")
        node = node[key]
    return node


def _extract_value(obj: Any, mode: str) -> Any:
    """
    Extract value from an object according to extraction mode.
    """
    if mode == "value":
        return obj

    if mode == "last":
        if isinstance(obj, deque):
            if len(obj) == 0:
                return None
            return obj[-1]
        if isinstance(obj, (list, tuple, np.ndarray)):
            if len(obj) == 0:
                return None
            return obj[-1]
        return obj

    if mode == "all":
        if isinstance(obj, deque):
            return list(obj)
        return obj

    raise ValueError(f"Unsupported extract mode: {mode}")


def _to_numpy(value: Any, dtype: Any) -> np.ndarray:
    """
    Convert input value to a numpy array with desired dtype.
    Scalars become shape () arrays.
    """
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)

    if isinstance(value, (list, tuple, deque)):
        return np.asarray(value, dtype=dtype)

    if isinstance(value, numbers.Number):
        return np.asarray(value, dtype=dtype)

    # fallback for objects that numpy can interpret
    return np.asarray(value, dtype=dtype)


def _validate_shape(arr: Optional[np.ndarray], expected_shape: Optional[tuple], field_name: str):
    if arr is None or expected_shape is None:
        return
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"Field '{field_name}' has shape {arr.shape}, expected {expected_shape}."
        )


# ============================================================
# Default split specifications for quadruped signals
# ============================================================
#
# These are used to turn vector-valued observations into more structured
# per-leg / per-sensor time series in the saved dataset.
#
# The mapping is:
#   base_field_name -> { part_name: index_spec }
# where:
#   - base_field_name matches a DataField.name (e.g. "joint_pos")
#   - part_name is an arbitrary string; the final key in the dataset
#     will be f"{base_field_name}/{part_name}"
#   - index_spec can be:
#       * int         (single component)
#       * slice       (contiguous block)
#       * list/tuple  (arbitrary indices)
#
quadruped_split_specs: Dict[str, Dict[str, Any]] = {
    # Joint-space signals: assume ordering
    # [FL_hip, FL_thigh, FL_calf,
    #  FR_hip, FR_thigh, FR_calf,
    #  RL_hip, RL_thigh, RL_calf,
    #  RR_hip, RR_thigh, RR_calf]
    "joint_pos": {
        "FL/hip": 0,    # HAA
        "FL/thigh": 1,  # HFE
        "FL/calf": 2,   # KFE
        "FR/hip": 3,    # HAA
        "FR/thigh": 4,  # HFE
        "FR/calf": 5,   # KFE
        "RL/hip": 6,    # HAA    
        "RL/thigh": 7,  # HFE   
        "RL/calf": 8,   # KFE
        "RR/hip": 9,    # HAA               
        "RR/thigh": 10,  # HFE
        "RR/calf": 11,   # KFE
    },
    "joint_vel": {
        "FL/hip": 0,    # HAA
        "FL/thigh": 1,  # HFE
        "FL/calf": 2,   # KFE
        "FR/hip": 3,    # HAA
        "FR/thigh": 4,  # HFE
        "FR/calf": 5,   # KFE
        "RL/hip": 6,    # HAA    
        "RL/thigh": 7,  # HFE   
        "RL/calf": 8,   # KFE
        "RR/hip": 9,    # HAA               
        "RR/thigh": 10,  # HFE
        "RR/calf": 11,   # KFE
    },
    # NOTE: the corresponding DataField should have name "torques"
    # to match the recorded feature_data structure.
    "torques": {
        "FL/hip": 0,    # HAA   
        "FL/thigh": 1,  # HFE
        "FL/calf": 2,   # KFE
        "FR/hip": 3,    # HAA
        "FR/thigh": 4,  # HFE
        "FR/calf": 5,   # KFE
        "RL/hip": 6,    # HAA    
        "RL/thigh": 7,  # HFE   
        "RL/calf": 8,   # KFE
        "RR/hip": 9,    # HAA               
        "RR/thigh": 10,  # HFE
        "RR/calf": 11,   # KFE
    },
    # IMU signals: each a 3D vector. The expected DataField names
    # (to match your feature_data) are "gyro" and "acc".
    "gyro": {
        "X": 0,
        "Y": 1,
        "Z": 2,
    },
    "acc": {
        "X": 0,
        "Y": 1,
        "Z": 2,
    },
    # Foot positions and velocities: assume stacking as
    # [FL_x, FL_y, FL_z, FR_x, FR_y, FR_z, RL_x, RL_y, RL_z, RR_x, RR_y, RR_z]
    "foot_pos": {
        "FL/x": 0,
        "FL/y": 1,
        "FL/z": 2,
        "FR/x": 3,
        "FR/y": 4,
        "FR/z": 5,
        "RL/x": 6,
        "RL/y": 7,
        "RL/z": 8,
        "RR/x": 9,
        "RR/y": 10,
        "RR/z": 11,
    },
    "foot_vel": {
        "FL/x": 0,
        "FL/y": 1,
        "FL/z": 2,
        "FR/x": 3,
        "FR/y": 4,
        "FR/z": 5,
        "RL/x": 6,
        "RL/y": 7,
        "RL/z": 8,
        "RR/x": 9,
        "RR/y": 10,
        "RR/z": 11,
    },
    # Ground reaction forces: same per-foot stacking convention as foot_pos/foot_vel.
    # Expected DataField name: "grf"
    "grf": {
        "FL/x": 0,
        "FL/y": 1,
        "FL/z": 2,
        "FR/x": 3,
        "FR/y": 4,
        "FR/z": 5,
        "RL/x": 6,
        "RL/y": 7,
        "RL/z": 8,
        "RR/x": 9,
        "RR/y": 10,
        "RR/z": 11,
    },
    # Binary contact state per foot. Expected DataField name: "contact_state"
    # Shape assumed (4,) ordered as [FL, FR, RL, RR].
    "contact_state": {
        "FL": 0,
        "FR": 1,
        "RL": 2,
        "RR": 3,
    },
    # Trunk wrench or force vector. Expected DataField name: "Trunk_Force"
    # Shape assumed (3,) ordered as [Fx, Fy, Fz].
    "Trunk_Force": {
        "X": 0,
        "Y": 1,
        "Z": 2,
    },
}


# ============================================================
# Storage backends
# ============================================================

class BaseBackend:
    def save_shard(self, shard_path: str, buffer_dict: Dict[str, List[np.ndarray]]) -> None:
        raise NotImplementedError

    def file_extension(self) -> str:
        raise NotImplementedError


class PickleBackend(BaseBackend):
    def __init__(self, gzip_compress: bool = False):
        self.gzip_compress = gzip_compress

    def file_extension(self) -> str:
        return ".pkl.gz" if self.gzip_compress else ".pkl"

    def save_shard(self, shard_path: str, buffer_dict: Dict[str, List[np.ndarray]]) -> None:
        payload = {}
        for k, values in buffer_dict.items():
            payload[k] = values

        if self.gzip_compress:
            with gzip.open(shard_path, "wb", compresslevel=4) as f:
                pkl.dump(payload, f, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            with open(shard_path, "wb") as f:
                pkl.dump(payload, f, protocol=pkl.HIGHEST_PROTOCOL)


class NumpyBackend(BaseBackend):
    """
    Saves one shard as .npz.
    Tries to stack arrays when shapes are consistent.
    Falls back to object arrays otherwise.
    """
    def file_extension(self) -> str:
        return ".npz"

    def save_shard(self, shard_path: str, buffer_dict: Dict[str, List[np.ndarray]]) -> None:
        save_dict = {}
        for k, values in buffer_dict.items():
            try:
                save_dict[k] = np.stack(values, axis=0)
            except Exception:
                save_dict[k] = np.array(values, dtype=object)
        np.savez_compressed(shard_path, **save_dict)


class H5Backend(BaseBackend):
    """
    Saves one shard as HDF5.
    Best when all samples for each field have same shape.
    """
    def file_extension(self) -> str:
        return ".h5"

    def save_shard(self, shard_path: str, buffer_dict: Dict[str, List[np.ndarray]]) -> None:
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is not installed. Please install h5py to use H5Backend.")

        with h5py.File(shard_path, "w") as f:
            for k, values in buffer_dict.items():
                try:
                    arr = np.stack(values, axis=0)
                    f.create_dataset(k, data=arr, compression="gzip")
                except Exception as e:
                    raise ValueError(
                        f"HDF5 backend requires consistent shapes for field '{k}'. "
                        f"Original error: {e}"
                    )


# ============================================================
# Dataset builder
# ============================================================

class FlexibleDatasetWriter:
    """
    Schema-driven dataset writer for proprioceptive data and ground truth.

    Supports:
      - configurable fields
      - nested dictionary path access
      - pickle / npz / h5 backends
      - buffered sharded saving
      - metadata export

    Example use cases:
      - save scalar proprioceptive features
      - save time windows from deque buffers
      - save multi-head labels (contact, GRF, slip, etc.)
    """

    def __init__(
        self,
        root_dir: str,
        fields: List[DataField],
        storage_format: str = "pickle",   # "pickle", "numpy", "h5"
        enable: bool = True,
        flush_every: int = 5000,
        max_buffer_items: int = 500_000,
        gzip_compress: bool = False,
        dataset_name: str = "dataset",
        strict: bool = True,
        split_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.enable = bool(enable)
        self.fields = fields
        self.flush_every = int(flush_every)
        self.max_buffer_items = int(max_buffer_items)
        # keep track of the user-provided base name and a timestamped variant
        self.base_dataset_name = dataset_name
        self.created_at = _now_timestamp()
        self.dataset_name = f"{dataset_name}_{self.created_at}"
        self.strict = strict
        # Optional specification to split certain 1D/vector fields
        # into multiple named sub-fields when saving. By default we
        # use the quadruped-specific conventions defined above.
        self.split_specs = split_specs if split_specs is not None else quadruped_split_specs

        # use the timestamped dataset name for the directory on disk
        self.directory = os.path.join(root_dir, self.dataset_name)
        os.makedirs(self.directory, exist_ok=True)

        self.backend = self._build_backend(storage_format, gzip_compress)
        self.storage_format = storage_format

        self._buffer = {field.name: [] for field in self.fields}
        self._n_samples = 0
        self._items_in_buffer = 0
        self._num_shards = 0

        self._save_metadata()

        if self.enable:
            print(f"[FlexibleDatasetWriter] Enabled")
            print(f"  directory      : {self.directory}")
            print(f"  storage_format : {self.storage_format}")
            print(f"  flush_every    : {self.flush_every}")

    # --------------------------------------------------------
    # setup
    # --------------------------------------------------------

    def _build_backend(self, storage_format: str, gzip_compress: bool) -> BaseBackend:
        storage_format = storage_format.lower()
        if storage_format == "pickle":
            return PickleBackend(gzip_compress=gzip_compress)
        if storage_format == "numpy":
            return NumpyBackend()
        if storage_format == "h5":
            return H5Backend()
        raise ValueError(f"Unsupported storage_format: {storage_format}")

    def _save_metadata(self):
        metadata = {
            "dataset_name": self.dataset_name,
            "base_dataset_name": getattr(self, "base_dataset_name", self.dataset_name),
            "storage_format": self.storage_format,
            "fields": [asdict(f) for f in self.fields],
            "created_at": self.created_at,
            "split_specs": self.split_specs,
        }
        meta_path = os.path.join(self.directory, "metadata.pkl")
        with open(meta_path, "wb") as f:
            pkl.dump(metadata, f, protocol=pkl.HIGHEST_PROTOCOL)

    # --------------------------------------------------------
    # internal
    # --------------------------------------------------------

    def _extract_field(self, source_dict: Dict[str, Any], field: DataField) -> Optional[np.ndarray]:
        try:
            raw = _resolve_nested_path(source_dict, field.source)
            value = _extract_value(raw, field.extract)
            arr = _to_numpy(value, field.dtype)
            _validate_shape(arr, field.shape, field.name)
            return arr
        except Exception as e:
            if field.required and self.strict:
                raise ValueError(f"Failed to extract required field '{field.name}': {e}") from e
            return None

    def _append_sample_to_buffer(self, sample: Dict[str, Optional[np.ndarray]]) -> None:
        for name, value in sample.items():
            if value is None:
                continue
            self._buffer[name].append(value)
            self._items_in_buffer += int(np.prod(value.shape)) if value.shape != () else 1

        self._n_samples += 1

    def _should_flush(self) -> bool:
        return (
            self._n_samples > 0 and
            (
                self._n_samples % self.flush_every == 0 or
                self._items_in_buffer >= self.max_buffer_items
            )
        )

    def _make_shard_path(self) -> str:
        shard_name = f"shard_{self._num_shards:06d}_{_now_timestamp()}{self.backend.file_extension()}"
        return os.path.join(self.directory, shard_name)

    def _reset_buffer(self):
        self._buffer = {field.name: [] for field in self.fields}
        self._items_in_buffer = 0

    def _apply_split_specs(self, buffer_dict: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        """
        Apply split specifications to selected vector fields so that, for example,
        a field like 'joint_pos' with shape (12,) per-sample can be saved as
        multiple time series, one per joint/leg.
        """
        if not self.split_specs:
            return buffer_dict

        # Start with a shallow copy so we don't mutate the original buffer
        new_dict: Dict[str, List[np.ndarray]] = {k: list(v) for k, v in buffer_dict.items()}

        for base_name, parts in self.split_specs.items():
            if base_name not in buffer_dict:
                continue

            values = buffer_dict[base_name]

            for part_name, idx_spec in parts.items():
                out_key = f"{base_name}/{part_name}"
                part_values: List[np.ndarray] = []
                for arr in values:
                    if arr is None:
                        part_values.append(None)
                        continue

                    # Support different index specifications:
                    # - int         -> single component (scalar)
                    # - slice       -> contiguous range
                    # - list/tuple  -> arbitrary indices
                    if isinstance(idx_spec, int):
                        sub = arr[idx_spec]
                    elif isinstance(idx_spec, slice):
                        sub = arr[idx_spec]
                    else:
                        sub = arr[idx_spec]

                    # ensure numpy array type to be consistent with rest of pipeline
                    part_values.append(np.asarray(sub, dtype=arr.dtype))

                new_dict[out_key] = part_values

            # Remove the original aggregated field to keep only the split series
            del new_dict[base_name]

        return new_dict

    # --------------------------------------------------------
    # public API
    # --------------------------------------------------------

    def append(self, feature_data: Dict[str, Any], gt_data: Dict[str, Any]) -> None:
        """
        Append one sample.

        Parameters
        ----------
        feature_data : dict
            Nested dict containing proprioceptive signals.
        gt_data : dict
            Nested dict containing labels / ground truth.
        """
        if not self.enable:
            return

        sample = {}

        for field in self.fields:
            source_root = feature_data if field.kind == "feature" else gt_data
            sample[field.name] = self._extract_field(source_root, field)

        self._append_sample_to_buffer(sample)

        if self._should_flush():
            self.save()

    def save(self) -> None:
        """
        Flush current buffer into one shard.
        """
        if not self.enable:
            return

        # do not save empty shard
        non_empty = any(len(v) > 0 for v in self._buffer.values())
        if not non_empty:
            return

        shard_path = self._make_shard_path()
        snapshot = self._apply_split_specs(self._buffer)
        self._reset_buffer()

        self.backend.save_shard(shard_path, snapshot)
        self._num_shards += 1

        print(f"[FlexibleDatasetWriter] Saved shard: {shard_path}")

    def close(self) -> None:
        self.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()