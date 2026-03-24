"""
Timeline Latent Visual Engine

Standalone interactive visualization for trained latent + theta sequences.
Designed for historical playback with a slider over dataset time (Mar 2023-Dec 2025).

Usage examples:
  python Generative_visualisation/latent_timeline_visual_engine.py
  python Generative_visualisation/latent_timeline_visual_engine.py \
      --latent AI_systems/latent_vectors.npy \
      --theta AI_systems/theta_pred.npy \
      --timestamps vae_model/data_full_2023_2025_check/timestamps.npy

Controls:
  - Slider: jump to any timestamp/index
  - Play/Pause button: autoplay
  - Left/Right arrows: step one frame
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, Slider


@dataclass
class SequenceData:
    latent: np.ndarray
    theta: np.ndarray
    timestamps: np.ndarray
    features: Optional[np.ndarray] = None


def _load_npy(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        return np.load(str(path))
    return None


def _to_unit(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = np.percentile(arr, 5)
    hi = np.percentile(arr, 95)
    if hi - lo <= 1e-8:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _fallback_theta_from_latent(latent: np.ndarray) -> np.ndarray:
    """Build deterministic pseudo-theta from latent if theta file is missing."""
    z = np.asarray(latent, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError("latent must be 2D")

    z0 = z[:, 0]
    z1 = z[:, 1] if z.shape[1] > 1 else z[:, 0]
    z2 = z[:, 2] if z.shape[1] > 2 else z[:, 0]
    z3 = z[:, 3] if z.shape[1] > 3 else z[:, 1]
    zn = np.linalg.norm(z, axis=1)

    t0 = _to_unit(np.abs(z0) + 0.35 * np.abs(z2))
    t1 = _to_unit(z1)
    t2 = _to_unit(np.abs(z2 - z3))
    t3 = _to_unit(np.abs(z0 - z1) + 0.2 * zn)
    t4 = _to_unit(np.abs(np.gradient(zn)) + 0.25 * np.abs(z3))

    return np.vstack([t0, t1, t2, t3, t4]).T.astype(np.float32)


def load_sequence_data(
    latent_path: Path,
    theta_path: Optional[Path],
    timestamps_path: Path,
    features_path: Optional[Path] = None,
) -> SequenceData:
    latent = _load_npy(latent_path)
    if latent is None:
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    timestamps = _load_npy(timestamps_path)
    if timestamps is None:
        raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")

    theta = None
    if theta_path is not None:
        theta = _load_npy(theta_path)

    features = None
    if features_path is not None:
        features = _load_npy(features_path)

    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim == 1:
        latent = latent.reshape(-1, 1)

    timestamps = np.asarray(timestamps)
    if timestamps.ndim != 1:
        timestamps = timestamps.reshape(-1)

    if theta is None:
        theta = _fallback_theta_from_latent(latent)
    else:
        theta = np.asarray(theta, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        if theta.shape[1] < 5:
            pad = np.zeros((theta.shape[0], 5 - theta.shape[1]), dtype=np.float32)
            theta = np.hstack([theta, pad])
        theta = theta[:, :5]

    if features is not None:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

    lengths = [len(latent), len(theta), len(timestamps)]
    if features is not None:
        lengths.append(len(features))

    n = min(lengths)
    if n <= 0:
        raise ValueError("No overlapping rows across latent/theta/timestamps")

    # Tail align to support context-window outputs.
    latent = latent[-n:]
    theta = theta[-n:]
    timestamps = timestamps[-n:]
    if features is not None:
        features = features[-n:]

    return SequenceData(latent=latent, theta=theta, timestamps=timestamps, features=features)


class TimelineVisualEngine:
    def __init__(self, data: SequenceData, autoplay_fps: float = 18.0, autoplay: bool = True):
        self.data = data
        self.n = len(data.timestamps)
        self.idx = 0
        self.playing = autoplay
        self.autoplay_interval_ms = int(1000.0 / max(1.0, autoplay_fps))

        self.day_to_index, self.day_labels = self._build_day_index(self.data.timestamps)
        self.n_days = len(self.day_to_index)
        self.day_ptr = 0

        self.theta_names = [
            "turbulence",
            "color_axis",
            "distortion",
            "fragmentation",
            "speed",
        ]

        self.fig = None
        self.ax_main = None
        self.ax_theta = None
        self.ax_latent = None
        self.ax_companion = None
        self.ax_slider = None
        self.ax_button = None

        self.slider = None
        self.play_button = None
        self.timer = None

        self.scatter = None
        self.cursor_theta = None
        self.cursor_latent = None
        self.cursor_companion = None
        self.info_text = None

        self.fear_greed_series = self._build_fear_greed_series()
        self.market_series = self._build_market_series()

        self.fg_cmap = LinearSegmentedColormap.from_list(
            "fear_greed",
            ["#5f0012", "#c1121f", "#e85d04", "#f4a261", "#7ec850", "#2a9d4b", "#0b6e3f"],
        )

        self.bg_cmap = LinearSegmentedColormap.from_list(
            "deep_sky",
            ["#03040a", "#0b1020", "#121b34", "#1a2450"],
        )

        self.rng = np.random.default_rng(7)
        self.base_n_particles = 1900
        self.base_positions = self._init_base_positions(self.base_n_particles)

        self._build_ui()
        self._draw_static_panels()
        self._render_day(0)

        if self.playing:
            self.play_button.label.set_text("Pause")
            self.timer.start()

    def _build_day_index(self, timestamps: np.ndarray) -> tuple[np.ndarray, list[str]]:
        dt = [datetime.fromtimestamp(int(ts), tz=timezone.utc) for ts in timestamps]
        day_keys = [d.strftime("%Y-%m-%d") for d in dt]

        day_to_index = []
        day_labels = []
        prev = None
        for i, k in enumerate(day_keys):
            if k != prev:
                day_to_index.append(i)
                day_labels.append(k)
                prev = k

        if not day_to_index:
            day_to_index = [0]
            day_labels = ["unknown"]

        return np.asarray(day_to_index, dtype=np.int32), day_labels

    def _build_fear_greed_series(self) -> np.ndarray:
        # Higher turbulence tends toward fear; color-axis trends toward greed.
        fg = 0.62 * self.data.theta[:, 1] + 0.38 * (1.0 - self.data.theta[:, 0])
        return np.clip(fg, 0.0, 1.0).astype(np.float32)

    def _build_market_series(self) -> np.ndarray:
        if self.data.features is not None and self.data.features.shape[1] > 0:
            base = self.data.features[:, 0]
        else:
            z0 = self.data.latent[:, 0]
            base = np.concatenate([[0.0], np.diff(z0)])
        pulse = np.cumsum(base.astype(np.float32))
        return _to_unit(pulse)

    def _init_base_positions(self, n_particles: int) -> np.ndarray:
        radius = self.rng.beta(1.2, 3.4, size=n_particles) * 1.14
        theta = self.rng.uniform(0.0, 2 * np.pi, size=n_particles)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Slightly denser core for a stronger focal point.
        core_n = max(120, int(0.17 * n_particles))
        core_r = self.rng.beta(1.0, 7.0, size=core_n) * 0.24
        core_t = self.rng.uniform(0.0, 2 * np.pi, size=core_n)
        x[:core_n] = core_r * np.cos(core_t)
        y[:core_n] = core_r * np.sin(core_t)

        return np.column_stack([x, y]).astype(np.float32)

    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(18.5, 10.0), facecolor="#03040a")
        gs = self.fig.add_gridspec(
            nrows=8,
            ncols=12,
            left=0.045,
            right=0.985,
            top=0.95,
            bottom=0.12,
            wspace=0.45,
            hspace=0.44,
        )

        self.ax_main = self.fig.add_subplot(gs[:, :8])
        self.ax_theta = self.fig.add_subplot(gs[:3, 8:])
        self.ax_latent = self.fig.add_subplot(gs[3:6, 8:])
        self.ax_companion = self.fig.add_subplot(gs[6:, 8:])

        self.ax_slider = self.fig.add_axes([0.08, 0.045, 0.76, 0.032], facecolor="#101935")
        self.ax_button = self.fig.add_axes([0.855, 0.038, 0.11, 0.045])

        self.slider = Slider(
            ax=self.ax_slider,
            label="Day",
            valmin=0,
            valmax=self.n_days - 1,
            valinit=0,
            valstep=1,
            color="#44b06a",
        )
        self.play_button = Button(self.ax_button, "Pause" if self.playing else "Play", color="#1e2f5d", hovercolor="#32539c")

        self.slider.on_changed(self._on_slider)
        self.play_button.on_clicked(self._on_play_pause)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        self.timer = self.fig.canvas.new_timer(interval=self.autoplay_interval_ms)
        self.timer.add_callback(self._on_timer)

        # Main visual panel style.
        self.ax_main.set_facecolor("#050812")
        self.ax_main.set_xlim(-1.3, 1.3)
        self.ax_main.set_ylim(-1.3, 1.3)
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        self.ax_main.set_title("Latent-Driven Particle Field", color="white", fontsize=14, pad=8)

        # Subplots style.
        for ax in [self.ax_theta, self.ax_latent, self.ax_companion]:
            ax.set_facecolor("#0b1328")
            ax.tick_params(colors="#d4d9ea", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#3a4d7a")

        self.info_text = self.ax_main.text(
            -1.28,
            1.23,
            "",
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            color="#eef3ff",
            bbox={"facecolor": "#122042", "alpha": 0.55, "edgecolor": "#4f6fb0", "pad": 6},
        )

    def _draw_static_panels(self) -> None:
        x = np.arange(self.n)

        # Theta trajectories.
        colors = ["#d62828", "#2a9d4b", "#f4a261", "#bc6c25", "#7ec850"]
        for i, name in enumerate(self.theta_names):
            self.ax_theta.plot(x, self.data.theta[:, i], lw=1.0, alpha=0.9, color=colors[i], label=name)

        self.ax_theta.set_title("Theta Controls Over Time", color="white", fontsize=11)
        self.ax_theta.set_ylim(-0.03, 1.03)
        self.ax_theta.set_ylabel("value", color="#d4d9ea", fontsize=8)
        self.ax_theta.legend(loc="upper right", fontsize=7, frameon=False)

        self.cursor_theta = self.ax_theta.axvline(0, color="#ffffff", lw=1.2, alpha=0.85)

        # Latent energy and directional change.
        znorm = np.linalg.norm(self.data.latent, axis=1)
        zspeed = np.concatenate([[0.0], np.linalg.norm(np.diff(self.data.latent, axis=0), axis=1)])
        z1 = self.data.latent[:, 0]

        self.ax_latent.plot(x, _to_unit(znorm), color="#66d9ef", lw=1.2, label="latent_norm")
        self.ax_latent.plot(x, _to_unit(zspeed), color="#ffcc66", lw=1.2, label="latent_speed")
        self.ax_latent.plot(x, _to_unit(z1), color="#f78c6c", lw=1.0, alpha=0.9, label="latent_axis_1")

        self.ax_latent.set_title("Latent Dynamics", color="white", fontsize=11)
        self.ax_latent.set_xlabel("index", color="#d4d9ea", fontsize=8)
        self.ax_latent.set_ylabel("normalized", color="#d4d9ea", fontsize=8)
        self.ax_latent.set_ylim(-0.03, 1.03)
        self.ax_latent.legend(loc="upper right", fontsize=7, frameon=False)

        self.cursor_latent = self.ax_latent.axvline(0, color="#ffffff", lw=1.2, alpha=0.85)

        # Companion panel: market proxy + fear/greed indicator.
        fg = self.fear_greed_series
        self.ax_companion.plot(x, self.market_series, color="#67b8ff", lw=1.1, alpha=0.95, label="btc_proxy")
        self.ax_companion.plot(x, fg, color="#b8ff7c", lw=1.2, alpha=0.95, label="fear_greed")
        self.ax_companion.fill_between(x, 0.0, fg, where=fg < 0.5, color="#d62828", alpha=0.18)
        self.ax_companion.fill_between(x, 0.0, fg, where=fg >= 0.5, color="#2a9d4b", alpha=0.18)

        self.ax_companion.set_title("BTC + Fear/Greed Companion", color="white", fontsize=11)
        self.ax_companion.set_xlabel("index", color="#d4d9ea", fontsize=8)
        self.ax_companion.set_ylabel("normalized", color="#d4d9ea", fontsize=8)
        self.ax_companion.set_ylim(-0.03, 1.03)
        self.ax_companion.legend(loc="upper right", fontsize=7, frameon=False)
        self.cursor_companion = self.ax_companion.axvline(0, color="#ffffff", lw=1.2, alpha=0.85)

    def _timestamp_str(self, unix_ts: int) -> str:
        dt = datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    def _particle_state(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        z = self.data.latent[idx]
        t = self.data.theta[idx]

        turbulence, color_axis, distortion, fragmentation, speed = [float(v) for v in t]

        p = self.base_positions.copy()
        n = len(p)

        # Procedural transform from latent + theta.
        phase = 0.12 * idx
        angle = np.arctan2(p[:, 1], p[:, 0])
        radius = np.linalg.norm(p, axis=1) + 1e-7

        twist = 0.25 + 1.1 * distortion + 0.15 * np.tanh(np.mean(z[: min(4, len(z))]))
        theta = angle + twist * radius + 0.22 * np.sin(phase + 2.5 * radius)

        swirl_x = radius * np.cos(theta)
        swirl_y = radius * np.sin(theta)

        jitter_amp = 0.002 + 0.045 * turbulence
        jitter = self.rng.normal(0.0, jitter_amp, size=(n, 2)).astype(np.float32)

        flow = np.array(
            [
                np.cos(0.11 * idx + 0.7 * speed),
                np.sin(0.09 * idx + 0.5 * speed),
            ],
            dtype=np.float32,
        )
        flow *= (0.01 + 0.05 * speed)

        p[:, 0] = swirl_x + jitter[:, 0] + flow[0]
        p[:, 1] = swirl_y + jitter[:, 1] + flow[1]

        # Fragmentation controls what fraction of points remain visible.
        keep_ratio = 0.46 + 0.52 * (1.0 - fragmentation)
        keep_n = max(220, int(keep_ratio * n))
        keep_idx = self.rng.choice(n, size=keep_n, replace=False)
        p = p[keep_idx]
        r_keep = radius[keep_idx]

        # Color from radial field + latent/color axis influence.
        fear_greed = float(self.fear_greed_series[idx])
        color_field = 0.70 * fear_greed + 0.30 * np.clip(1.0 - r_keep + 0.12 * np.sin(phase + 8.0 * r_keep), 0.0, 1.0)
        color_field = np.clip(color_field, 0.0, 1.0)

        # Point sizes from speed+turbulence.
        size_base = 2.5 + 22.0 * speed + 14.0 * turbulence
        sizes = np.clip(size_base * (0.45 + 0.9 * (1.0 - r_keep)), 2.0, 60.0)

        colors = self.fg_cmap(color_field)
        colors[:, 3] = np.clip(0.22 + 0.65 * (1.0 - r_keep), 0.15, 0.9)

        return p, sizes, colors, fear_greed

    def _render_day(self, day_idx: int) -> None:
        day_idx = int(np.clip(day_idx, 0, self.n_days - 1))
        self.day_ptr = day_idx
        idx = int(self.day_to_index[day_idx])
        self._render_index(idx)

    def _render_index(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, self.n - 1))
        self.idx = idx

        p, sizes, colors, fear_greed = self._particle_state(idx)

        self.ax_main.clear()
        self.ax_main.set_facecolor("#050812")
        self.ax_main.set_xlim(-1.3, 1.3)
        self.ax_main.set_ylim(-1.3, 1.3)
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        self.ax_main.set_title("Latent-Driven Particle Field", color="white", fontsize=14, pad=8)

        # Subtle atmospheric background.
        bg = np.outer(np.linspace(0, 1, 220), np.ones(220))
        self.ax_main.imshow(bg, extent=[-1.3, 1.3, -1.3, 1.3], origin="lower", cmap=self.bg_cmap, alpha=0.45)

        # Soft center glow.
        for r, a in [(0.42, 0.04), (0.28, 0.06), (0.14, 0.09)]:
            circ = plt.Circle((0.0, 0.0), radius=r, color="#ffd27f", alpha=a, ec="none")
            self.ax_main.add_patch(circ)

        self.scatter = self.ax_main.scatter(
            p[:, 0],
            p[:, 1],
            s=sizes,
            c=colors,
            edgecolors="none",
            zorder=4,
        )

        z = self.data.latent[idx]
        t = self.data.theta[idx]
        ts = self._timestamp_str(int(self.data.timestamps[idx]))

        mood = "GREED" if fear_greed >= 0.5 else "FEAR"
        info = (
            f"time  : {ts}\n"
            f"day   : {self.day_labels[self.day_ptr]} ({self.day_ptr+1}/{self.n_days})\n"
            f"index : {idx+1}/{self.n}\n"
            f"z||   : {np.linalg.norm(z):.4f}\n"
            f"z0/z1 : {z[0]:+.4f} / {z[1] if len(z)>1 else 0:+.4f}\n"
            f"mood  : {mood} ({fear_greed:.3f})\n"
            "\n"
            f"{self.theta_names[0]:<13}: {t[0]:.3f}\n"
            f"{self.theta_names[1]:<13}: {t[1]:.3f}\n"
            f"{self.theta_names[2]:<13}: {t[2]:.3f}\n"
            f"{self.theta_names[3]:<13}: {t[3]:.3f}\n"
            f"{self.theta_names[4]:<13}: {t[4]:.3f}\n"
        )
        self.info_text = self.ax_main.text(
            -1.28,
            1.23,
            info,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            color="#eef3ff",
            bbox={"facecolor": "#122042", "alpha": 0.55, "edgecolor": "#4f6fb0", "pad": 6},
            zorder=10,
        )

        self.cursor_theta.set_xdata([idx, idx])
        self.cursor_latent.set_xdata([idx, idx])
        self.cursor_companion.set_xdata([idx, idx])

        self.fig.canvas.draw_idle()

    def _on_slider(self, val: float) -> None:
        self._render_day(int(val))

    def _on_play_pause(self, _event) -> None:
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_timer(self) -> None:
        if not self.playing:
            return

        nxt = self.day_ptr + 1
        if nxt >= self.n_days:
            self.playing = False
            self.play_button.label.set_text("Play")
            self.timer.stop()
            return

        self.slider.set_val(nxt)

    def _on_key_press(self, event) -> None:
        if event.key == "right":
            self.slider.set_val(min(self.n_days - 1, self.day_ptr + 1))
        elif event.key == "left":
            self.slider.set_val(max(0, self.day_ptr - 1))

    def show(self) -> None:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive latent timeline visual engine")
    parser.add_argument(
        "--latent",
        default="AI_systems/latent_vectors.npy",
        help="Path to latent vectors .npy",
    )
    parser.add_argument(
        "--theta",
        default="AI_systems/theta_pred.npy",
        help="Path to theta predictions .npy (optional; fallback generated if missing)",
    )
    parser.add_argument(
        "--timestamps",
        default="vae_model/data_full_2023_2025_check/timestamps.npy",
        help="Path to timestamps .npy",
    )
    parser.add_argument(
        "--features",
        default="vae_model/data_full_2023_2025_check/full_features.npy",
        help="Path to full feature matrix .npy (optional, for companion panel market proxy)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Autoplay FPS",
    )
    parser.add_argument(
        "--no-autoplay",
        action="store_true",
        help="Disable autoplay at startup",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    latent_path = Path(args.latent)
    theta_path = Path(args.theta)
    timestamps_path = Path(args.timestamps)
    features_path = Path(args.features)

    data = load_sequence_data(
        latent_path=latent_path,
        theta_path=theta_path if theta_path.exists() else None,
        timestamps_path=timestamps_path,
        features_path=features_path if features_path.exists() else None,
    )

    print(f"[✓] Loaded sequence rows: {len(data.timestamps)}")
    print(f"    latent shape : {data.latent.shape}")
    print(f"    theta shape  : {data.theta.shape}")

    engine = TimelineVisualEngine(data=data, autoplay_fps=args.fps, autoplay=not args.no_autoplay)
    engine.show()


if __name__ == "__main__":
    main()
