#Usage: python3 display_1D_data.py
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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
    # default: try numpy first, then pickle
    return "numpy"


def load_series_dict(path: Path) -> Dict[str, np.ndarray]:
    """
    Load a file that contains one or more time series.

    - NumPy:
        * .npz: each key is treated as a separate series
        * .npy: if it stores a dict, each key is a series; otherwise one series
    - Pickle:
        * if it stores a dict, each key is a series; otherwise one series
    - HDF5:
        * each dataset directly under the root group is a series

    If a series is multi-dimensional (e.g. shape (T, D)), each column is split
    into its own 1D series with names like "key[0]", "key[1]", ...
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

    def add_series_from_array(base_key: str, arr: np.ndarray) -> None:
        # 1D array → single series
        if arr.ndim == 1:
            series[base_key] = arr
            return

        # Higher dimensional: flatten all non-time dimensions into channels
        if arr.ndim >= 2:
            t = arr.shape[0]
            rest = int(np.prod(arr.shape[1:]))
            flat = arr.reshape(t, rest)
            for idx in range(rest):
                series[f"{base_key}[{idx}]"] = flat[:, idx]

    if ftype == "numpy":
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=True)
            for key in data.files:
                a = to_array(data[key])
                if a is not None:
                    add_series_from_array(str(key), a)
        else:  # .npy or unknown
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, dict):
                for key, value in arr.items():
                    a = to_array(value)
                    if a is not None:
                        add_series_from_array(str(key), a)
            else:
                a = to_array(arr)
                if a is not None:
                    add_series_from_array("data", a)
    elif ftype == "pickle":
        if pickle is None:
            raise RuntimeError("pickle module not available.")
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for key, value in obj.items():
                a = to_array(value)
                if a is not None:
                    add_series_from_array(str(key), a)
        else:
            a = to_array(obj)
            if a is not None:
                add_series_from_array("data", a)
    else:  # "h5"
        if h5py is None:
            raise RuntimeError("h5py is not installed. Install with `pip install h5py`.")
        with h5py.File(path, "r") as f:
            for name, obj in f.items():
                if isinstance(obj, h5py.Dataset):
                    a = np.asarray(obj[()])
                    add_series_from_array(str(name), a)

    if not series:
        raise ValueError(f"No time series found in file {path}")

    return series


class SlidingWindowViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("1D Dataset Sliding Window Viewer")
        self.geometry("1000x700")

        # key: "file:key" -> 1D array
        self.series: Dict[str, np.ndarray] = {}
        self.visible: Dict[str, tk.BooleanVar] = {}

        self.window_size = 500
        self.current_start = 0
        self.max_length = 0

        # Play/animation state
        self.playing = False
        self._play_job: str | None = None

        self._build_ui()

    # UI ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Top control frame
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        load_btn = tk.Button(top_frame, text="Load Files", command=self.load_files)
        load_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="Window size:").pack(side=tk.LEFT, padx=(15, 0))
        self.window_size_var = tk.IntVar(value=self.window_size)
        window_entry = tk.Entry(top_frame, textvariable=self.window_size_var, width=8)
        window_entry.pack(side=tk.LEFT, padx=5)

        apply_window_btn = tk.Button(
            top_frame, text="Apply", command=self.apply_window_size
        )
        apply_window_btn.pack(side=tk.LEFT, padx=5)

        # Navigation frame
        nav_frame = tk.Frame(self)
        nav_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.start_label = tk.Label(nav_frame, text="Start: 0")
        self.start_label.pack(side=tk.LEFT, padx=(0, 10))

        self.play_button = tk.Button(nav_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)

        prev_btn = tk.Button(nav_frame, text="<< Prev", command=self.prev_window)
        prev_btn.pack(side=tk.LEFT, padx=5)

        next_btn = tk.Button(nav_frame, text="Next >>", command=self.next_window)
        next_btn.pack(side=tk.LEFT, padx=5)

        self.position_scale = tk.Scale(
            nav_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            length=400,
            command=self.on_scale_change,
        )
        self.position_scale.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # Main plot frame
        plot_frame = tk.Frame(self)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Index")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right-side toggle area with scrollable frame
        self.toggle_container = tk.Frame(plot_frame)
        self.toggle_container.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        self.toggle_label = tk.Label(self.toggle_container, text="Datasets")
        self.toggle_label.pack(anchor="n", pady=(0, 5))

        # Canvas + scrollbar to allow many options
        self.toggle_canvas = tk.Canvas(self.toggle_container, borderwidth=0, highlightthickness=0)
        self.toggle_scrollbar = tk.Scrollbar(
            self.toggle_container, orient="vertical", command=self.toggle_canvas.yview
        )
        self.toggle_canvas.configure(yscrollcommand=self.toggle_scrollbar.set)

        self.toggle_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.toggle_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.toggle_frame = tk.Frame(self.toggle_canvas)
        self.toggle_canvas.create_window((0, 0), window=self.toggle_frame, anchor="nw")

        # Ensure scrollregion updates when contents change
        def _on_frame_configure(event: object) -> None:  # type: ignore[override]
            self.toggle_canvas.configure(scrollregion=self.toggle_canvas.bbox("all"))

        self.toggle_frame.bind("<Configure>", _on_frame_configure)

    # Dataset management ---------------------------------------------------
    def load_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select 1D dataset files",
            filetypes=[
                (
                    "Dataset files",
                    "*.npy *.npz *.h5 *.hdf5 *.pkl *.pickle",
                ),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        new_series: List[Tuple[str, np.ndarray]] = []
        errors: List[str] = []

        for p in paths:
            path = Path(p)
            try:
                series_dict = load_series_dict(path)
                for key, arr in series_dict.items():
                    # Use only the series key as label (no file name prefix)
                    label = str(key)
                    new_series.append((label, arr))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{path}: {e}")

        if errors:
            messagebox.showwarning(
                "Some files failed to load",
                "\n".join(errors),
            )

        if not new_series:
            return

        for label, arr in new_series:
            self.series[label] = arr

        self._update_max_length()
        self._rebuild_toggles()
        self.current_start = 0
        self._update_scale_limits()
        self.redraw_plot()

    def _update_max_length(self) -> None:
        if not self.series:
            self.max_length = 0
        else:
            self.max_length = max(len(a) for a in self.series.values())

    def _rebuild_toggles(self) -> None:
        for child in self.toggle_frame.winfo_children():
            child.destroy()

        self.visible.clear()
        for name in sorted(self.series.keys()):
            # Start with all series disabled (not shown)
            var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(
                self.toggle_frame,
                text=name,
                variable=var,
                command=self.redraw_plot,
                anchor="w",
                justify="left",
            )
            chk.pack(fill=tk.X, anchor="w")
            self.visible[name] = var

    # Window navigation ----------------------------------------------------
    def apply_window_size(self) -> None:
        try:
            new_size = int(self.window_size_var.get())
            if new_size <= 0:
                raise ValueError
        except Exception:  # noqa: BLE001
            messagebox.showerror("Invalid value", "Window size must be a positive integer.")
            self.window_size_var.set(self.window_size)
            return

        self.window_size = new_size
        self.current_start = 0
        self._update_scale_limits()
        self.redraw_plot()

    def _update_scale_limits(self) -> None:
        if self.max_length <= self.window_size:
            self.position_scale.config(from_=0, to=0)
        else:
            self.position_scale.config(from_=0, to=self.max_length - self.window_size)
        self.position_scale.set(self.current_start)
        self.start_label.config(text=f"Start: {self.current_start}")

    def prev_window(self) -> None:
        if self.current_start <= 0:
            return
        self.current_start = max(0, self.current_start - self.window_size)
        self._update_scale_limits()
        self.redraw_plot()

    def next_window(self) -> None:
        if self.max_length <= self.window_size:
            return
        self.current_start = min(
            self.max_length - self.window_size,
            self.current_start + self.window_size,
        )
        self._update_scale_limits()
        self.redraw_plot()

    def on_scale_change(self, value: str) -> None:
        try:
            self.current_start = int(float(value))
        except ValueError:
            return
        self.start_label.config(text=f"Start: {self.current_start}")
        self.redraw_plot()

    # Plotting -------------------------------------------------------------
    def redraw_plot(self) -> None:
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xlabel("Index")
        self.ax.set_ylabel("Value")

        if not self.series:
            self.canvas.draw()
            return

        start = self.current_start
        end = start + self.window_size

        for name, arr in sorted(self.series.items()):
            if not self.visible.get(name, tk.BooleanVar(value=True)).get():
                continue
            segment = arr[start:end]
            if segment.size == 0:
                continue
            x = np.arange(start, start + len(segment))
            self.ax.plot(x, segment, label=name)

        self.ax.legend(loc="upper right", fontsize="small")
        self.canvas.draw()

    # Animation / play ------------------------------------------------------
    def toggle_play(self) -> None:
        if self.playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self) -> None:
        if not self.series or self.max_length <= self.window_size:
            return
        self.playing = True
        self.play_button.config(text="Pause")
        # Start from current position and advance window-by-window
        self._schedule_next_frame()

    def stop_play(self) -> None:
        self.playing = False
        self.play_button.config(text="Play")
        if self._play_job is not None:
            try:
                self.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None

    def _schedule_next_frame(self) -> None:
        if not self.playing:
            return

        # Advance by one sample each step
        step = 1
        if self.max_length <= self.window_size:
            self.stop_play()
            return

        next_start = self.current_start + step
        # If advancing would go past the end, show the last window and stop
        if next_start > self.max_length - self.window_size:
            next_start = self.max_length - self.window_size
            self.current_start = next_start
            self._update_scale_limits()
            self.redraw_plot()
            self.stop_play()
            return

        self.current_start = next_start
        self._update_scale_limits()
        self.redraw_plot()

        # Schedule next frame (in milliseconds)
        self._play_job = str(self.after(30, self._schedule_next_frame))


def main() -> None:
    app = SlidingWindowViewer()
    app.mainloop()


if __name__ == "__main__":
    main()

