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
import time
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
        self.playing = autoplay
        self.autoplay_fps = float(max(1.0, autoplay_fps))
        self.timer_interval_ms = int(1000.0 / max(24.0, min(60.0, self.autoplay_fps * 2.0)))
        self.playback_speed = float(max(0.25, min(3.0, self.autoplay_fps / 10.0)))

        self.month_to_index = self._build_month_index(self.data.timestamps)
        self.n_months = len(self.month_to_index)
        self.month_ptr = 0
        self.current_month_float = 0.0
        self.last_timer_ts = None
        self.rotation_phase = 0.0

        # Faster timeline progression while keeping cinematic pacing.
        self.months_per_second = 0.04032 * self.playback_speed
        self.predicted_total_seconds = 0.0 if self.n_months <= 1 else (self.n_months - 1) / self.months_per_second
        self.play_elapsed_seconds = 0.0

        self.fig = None
        self.ax_main = None
        self.ax_slider = None
        self.ax_button = None

        self.slider = None
        self.play_button = None
        self.timer = None
        self._syncing_slider = False

        self.scatter = None
        self.glow_patches = []
        self.hud_text = None
        self.theta_axes = []
        self.theta_lines = []
        self.theta_markers = []
        self.theta_explainer_ax = None

        self.fear_greed_series = self._build_fear_greed_series()
        self.month_theta = self.data.theta[self.month_to_index, :5].astype(np.float32)

        self.fg_cmap = LinearSegmentedColormap.from_list(
            "fear_greed",
            ["#7a1022", "#c8342a", "#ff9448", "#ffe08a", "#a5e676", "#69c77a", "#66b6ff"],
        )

        self.bg_cmap = LinearSegmentedColormap.from_list(
            "deep_space",
            ["#02030a", "#040816", "#0a1230", "#13244a"],
        )

        self.rng = np.random.default_rng(7)
        self.base_n_particles = 1900
        self.base_positions = self._init_base_positions(self.base_n_particles)
        self.base_depth = self.rng.uniform(0.0, 1.0, size=self.base_n_particles).astype(np.float32)
        self.travel_seed = self.rng.uniform(0.0, 2.0 * np.pi, size=self.base_n_particles).astype(np.float32)

        phase_seed = self.rng.uniform(0.0, 2.0 * np.pi, size=self.base_n_particles).astype(np.float32)
        dir_seed = self.rng.uniform(0.0, 2.0 * np.pi, size=self.base_n_particles).astype(np.float32)
        self.jitter_phase_seed = phase_seed
        self.jitter_dir = np.column_stack([np.cos(dir_seed), np.sin(dir_seed)]).astype(np.float32)
        self.current_keep_idx = np.arange(self.base_n_particles, dtype=np.int32)

        self._build_ui()
        self._set_keep_mask_for_month(0.0)
        self._render_month(0, immediate=True)

        if self.playing:
            self.play_button.label.set_text("Pause")
            self.last_timer_ts = time.perf_counter()
            self.timer.start()

    def _build_month_index(self, timestamps: np.ndarray) -> np.ndarray:
        dt = [datetime.fromtimestamp(int(ts), tz=timezone.utc) for ts in timestamps]
        month_keys = [d.strftime("%Y-%m") for d in dt]

        month_to_index = []
        prev = None
        for i, k in enumerate(month_keys):
            if k != prev:
                month_to_index.append(i)
                prev = k

        if not month_to_index:
            month_to_index = [0]

        return np.asarray(month_to_index, dtype=np.int32)

    def _build_fear_greed_series(self) -> np.ndarray:
        # Higher turbulence tends toward fear; color-axis trends toward greed.
        fg = 0.62 * self.data.theta[:, 1] + 0.38 * (1.0 - self.data.theta[:, 0])
        return np.clip(fg, 0.0, 1.0).astype(np.float32)

    def _init_base_positions(self, n_particles: int) -> np.ndarray:
        # Milky-way style disk: elongated central bulge with subtle spiral structure.
        arm_id = self.rng.integers(0, 4, size=n_particles)
        radius = self.rng.beta(1.1, 2.0, size=n_particles) * 3.2
        arm_phase = arm_id * (2 * np.pi / 4.0)
        theta = arm_phase + 1.45 * radius + self.rng.normal(0.0, 0.26, size=n_particles)

        x = radius * np.cos(theta)
        y = 0.34 * radius * np.sin(theta)

        # Dense galactic bulge at center.
        core_n = max(200, int(0.22 * n_particles))
        core_r = self.rng.beta(1.0, 6.5, size=core_n) * 0.95
        core_t = self.rng.uniform(0.0, 2 * np.pi, size=core_n)
        x[:core_n] = core_r * np.cos(core_t)
        y[:core_n] = 0.45 * core_r * np.sin(core_t)

        return np.column_stack([x, y]).astype(np.float32)

    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(15.5, 9.2), facecolor="#02030a")
        gs = self.fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[4.65, 1.55],
            left=0.04,
            right=0.985,
            top=0.95,
            bottom=0.12,
            wspace=0.06,
        )

        self.ax_main = self.fig.add_subplot(gs[0, 0])
        theta_gs = gs[0, 1].subgridspec(6, 1, height_ratios=[1, 1, 1, 1, 1, 0.9], hspace=0.17)

        self.ax_slider = self.fig.add_axes([0.08, 0.045, 0.74, 0.032], facecolor="#101935")
        self.ax_button = self.fig.add_axes([0.835, 0.038, 0.13, 0.045])

        self.slider = Slider(
            ax=self.ax_slider,
            label="Month",
            valmin=0,
            valmax=self.n_months - 1,
            valinit=0,
            color="#58c6ff",
        )
        self.play_button = Button(self.ax_button, "Pause" if self.playing else "Play", color="#1e2f5d", hovercolor="#32539c")

        # Slider visual polish for a cleaner, modern look.
        self.ax_slider.set_facecolor("#0e1833")
        for spine in self.ax_slider.spines.values():
            spine.set_color("#34507d")
        self.slider.label.set_color("#dfefff")
        self.slider.valtext.set_color("#9fc0ea")
        self.slider.poly.set_facecolor("#58c6ff")
        self.slider.vline.set_color("#b5ecff")

        self.slider.on_changed(self._on_slider)
        self.play_button.on_clicked(self._on_play_pause)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Frequent timer ticks are used for interpolation; autoplay speed controls day cadence.
        self.timer = self.fig.canvas.new_timer(interval=self.timer_interval_ms)
        self.timer.add_callback(self._on_timer)

        # Main visual panel style.
        self.ax_main.set_facecolor("#02030a")
        self.ax_main.set_xlim(-3.3, 3.3)
        self.ax_main.set_ylim(-3.3, 3.3)
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        self.ax_main.set_title("Milkyway Latent Flight", color="#eaf0ff", fontsize=14, pad=8)

        # Static background and glow are drawn once and reused.
        bg = np.outer(np.linspace(0, 1, 220), np.ones(220))
        self.ax_main.imshow(
            bg,
            extent=[-3.3, 3.3, -3.3, 3.3],
            origin="lower",
            cmap=self.bg_cmap,
            alpha=0.80,
            zorder=1,
        )

        # Central galactic glow.
        for r, a in [(0.95, 0.024), (0.62, 0.036), (0.35, 0.064)]:
            circ = plt.Circle((0.0, 0.0), radius=r, color="#ffd89a", alpha=a, ec="none", zorder=2)
            self.ax_main.add_patch(circ)
            self.glow_patches.append(circ)

        # Distant starfield layer.
        n_bg = 480
        bg_x = self.rng.uniform(-3.3, 3.3, size=n_bg)
        bg_y = self.rng.uniform(-3.3, 3.3, size=n_bg)
        bg_s = self.rng.uniform(2.0, 10.0, size=n_bg)
        bg_c = np.ones((n_bg, 4), dtype=np.float32)
        bg_c[:, :3] = np.array([0.90, 0.94, 1.0], dtype=np.float32)
        bg_c[:, 3] = self.rng.uniform(0.05, 0.225, size=n_bg)
        self.ax_main.scatter(bg_x, bg_y, s=bg_s, c=bg_c, edgecolors="none", zorder=2)

        self.scatter = self.ax_main.scatter(
            np.zeros(self.base_n_particles, dtype=np.float32),
            np.zeros(self.base_n_particles, dtype=np.float32),
            s=np.full(self.base_n_particles, 12.5, dtype=np.float32),
            c=np.tile(np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32), (self.base_n_particles, 1)),
            edgecolors="none",
            marker="o",
            zorder=4,
        )

        self.hud_text = self.ax_main.text(
            -3.18,
            3.12,
            "",
            va="top",
            ha="left",
            fontsize=10,
            color="#eef4ff",
            family="monospace",
            bbox={"facecolor": "#0a1224", "alpha": 0.76, "edgecolor": "#2a3f6d", "pad": 6},
            zorder=10,
        )

        # Theta panels: show each signal over month-index time with a live cursor marker.
        theta_titles = [
            "θ0 Turbulence",
            "θ1 Trend Bias",
            "θ2 Distortion",
            "θ3 Fragmentation",
            "θ4 Velocity",
        ]
        theta_desc = [
            "market agitation / volatility",
            "directional market pressure",
            "regime-shape irregularity",
            "cohesion vs sparse breakup",
            "pace of latent-state change",
        ]
        x_month = np.arange(self.n_months, dtype=np.float32)

        self.theta_axes = []
        self.theta_lines = []
        self.theta_markers = []

        for i in range(5):
            ax = self.fig.add_subplot(theta_gs[i, 0])
            ax.set_facecolor("#070d1d")
            for s in ax.spines.values():
                s.set_color("#314a77")

            y = self.month_theta[:, i] if self.n_months > 0 else np.zeros(1, dtype=np.float32)
            line, = ax.plot(x_month, y, color="#7fd1ff", lw=1.25, alpha=0.95)
            marker, = ax.plot([0.0], [float(y[0]) if len(y) else 0.0], marker="o", ms=5.0, color="#ffd27c", lw=0)

            ax.set_xlim(0, max(1, self.n_months - 1))
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.18, color="#7ca0d4", linewidth=0.55)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(axis="y", colors="#a9c3e8", labelsize=7)
            ax.tick_params(axis="x", colors="#8fb2df", labelsize=7)
            ax.set_ylabel(f"θ{i}", color="#b9cff0", fontsize=7, labelpad=2)
            ax.set_title(f"{theta_titles[i]}  -  {theta_desc[i]}", color="#dbe9ff", fontsize=8, loc="left", pad=2)

            if i < 4:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Month Index", color="#9cb9e0", fontsize=8)

            self.theta_axes.append(ax)
            self.theta_lines.append(line)
            self.theta_markers.append(marker)

        self.theta_explainer_ax = self.fig.add_subplot(theta_gs[5, 0])
        self.theta_explainer_ax.set_facecolor("#060c1a")
        self.theta_explainer_ax.set_xticks([])
        self.theta_explainer_ax.set_yticks([])
        for s in self.theta_explainer_ax.spines.values():
            s.set_color("#314a77")
        self.theta_explainer_ax.text(
            0.02,
            0.92,
            "Theta Meanings (normalized 0-1)",
            transform=self.theta_explainer_ax.transAxes,
            va="top",
            ha="left",
            color="#e7f0ff",
            fontsize=8.5,
            fontweight="bold",
        )
        self.theta_explainer_ax.text(
            0.02,
            0.74,
            "θ0 Turbulence: calm -> choppy\n"
            "θ1 Trend Bias: bearish -> bullish\n"
            "θ2 Distortion: pattern regularity -> regime-shift irregularity\n"
            "θ3 Fragmentation: cohesive structure -> broken/scattered structure\n"
            "θ4 Velocity: slow evolution -> fast latent-state change",
            transform=self.theta_explainer_ax.transAxes,
            va="top",
            ha="left",
            color="#b7cdef",
            fontsize=8,
            linespacing=1.35,
        )

    def _state_at_month(self, month_float: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        month_float = float(np.clip(month_float, 0.0, self.n_months - 1))
        m0 = int(np.floor(month_float))
        m1 = min(self.n_months - 1, m0 + 1)
        a = month_float - m0

        idx0 = int(self.month_to_index[m0])
        idx1 = int(self.month_to_index[m1])

        z0 = self.data.latent[idx0]
        z1 = self.data.latent[idx1]
        t0 = self.data.theta[idx0]
        t1 = self.data.theta[idx1]

        z = ((1.0 - a) * z0 + a * z1).astype(np.float32)
        t = ((1.0 - a) * t0 + a * t1).astype(np.float32)
        fear_greed = float((1.0 - a) * self.fear_greed_series[idx0] + a * self.fear_greed_series[idx1])

        return z, t, fear_greed, np.array([idx0, idx1, a], dtype=np.float32)

    def _compute_keep_indices(self, fragmentation: float, seed_value: int) -> np.ndarray:
        # Wider sparsity range for a freer, less uniformly dense field.
        keep_ratio = 0.20 + 0.70 * (1.0 - float(fragmentation))
        keep_n = max(220, int(keep_ratio * self.base_n_particles))
        local_rng = np.random.default_rng(int(seed_value) + 1729)
        return np.sort(local_rng.choice(self.base_n_particles, size=keep_n, replace=False)).astype(np.int32)

    def _set_keep_mask_for_month(self, month_float: float) -> None:
        _z, t, _fg, idx_info = self._state_at_month(month_float)
        idx0 = int(idx_info[0])
        idx1 = int(idx_info[1])
        a = float(idx_info[2])
        frag = float(t[3])
        seed_value = int(round((1.0 - a) * idx0 + a * idx1))
        self.current_keep_idx = self._compute_keep_indices(frag, seed_value)

    def _particle_state(self, month_float: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z, t, fear_greed, idx_info = self._state_at_month(month_float)
        idx0 = int(idx_info[0])
        idx1 = int(idx_info[1])
        a = float(idx_info[2])

        turbulence = float(t[0])
        distortion = float(t[2])
        fragmentation = float(t[3])
        speed = float(t[4])

        p = self.base_positions.copy()

        # Center-based spiral transform.
        phase_idx = (1.0 - a) * idx0 + a * idx1
        phase = self.rotation_phase

        radius = np.linalg.norm(p, axis=1) + 1e-7
        angle = np.arctan2(p[:, 1], p[:, 0])

        twist = 0.65 + 1.35 * distortion
        # Direct angular term ensures clear one-direction rotation over time.
        angular_velocity = 0.144 + 0.3456 * speed + 0.1152 * distortion
        # Static radial warp keeps texture without introducing directional wobble.
        distortion_warp = 0.238 * distortion * np.sin(3.0 * angle + 0.75 * phase)
        spiral_theta = angle + twist * radius + angular_velocity * phase + 0.05 * np.sin(2.0 * radius) + distortion_warp
        breathing = 1.0 + 0.02 * np.sin(phase + 1.1 * radius)

        swirl_x = breathing * radius * np.cos(spiral_theta)
        swirl_y = breathing * radius * np.sin(spiral_theta)

        drift_x = 0.08 * np.tanh(z[0] if len(z) > 0 else 0.0)
        drift_y = 0.06 * np.tanh(z[1] if len(z) > 1 else 0.0)

        jitter_amp = 0.00025 + 0.0018 * turbulence
        jitter_scalar = np.sin(self.jitter_phase_seed + 0.055 * phase_idx).astype(np.float32)
        jitter = self.jitter_dir * (jitter_amp * jitter_scalar).reshape(-1, 1)

        p[:, 0] = swirl_x + drift_x + jitter[:, 0]
        p[:, 1] = swirl_y + drift_y + jitter[:, 1]

        # Keep points inside viewport with soft compression (prevents drifting out of frame).
        p[:, 0] = 3.05 * np.tanh(p[:, 0] / 2.25)
        p[:, 1] = 3.05 * np.tanh(p[:, 1] / 2.25)

        # Keep mask is fixed during a transition to avoid point popping.
        keep_idx = self.current_keep_idx
        p = p[keep_idx]
        radius_keep = np.linalg.norm(p, axis=1)
        depth_keep = np.clip(radius_keep / 3.2, 0.0, 1.0)

        # Higher fragmentation visibly breaks the field apart by pushing points outward.
        unit_vec = p / np.clip(radius_keep.reshape(-1, 1), 1e-6, None)
        frag_push = (fragmentation ** 1.15) * (0.07 + 0.294 * depth_keep)
        p += unit_vec * frag_push.reshape(-1, 1)
        p[:, 0] = 3.05 * np.tanh(p[:, 0] / 2.25)
        p[:, 1] = 3.05 * np.tanh(p[:, 1] / 2.25)
        radius_keep = np.linalg.norm(p, axis=1)
        depth_keep = np.clip(radius_keep / 3.2, 0.0, 1.0)

        # Color from radial field + latent/color axis influence.
        color_field = 0.68 * fear_greed + 0.32 * np.clip(1.0 - depth_keep + 0.18 * np.sin(phase + 3.5 * depth_keep), 0.0, 1.0)
        color_field = np.clip(color_field, 0.0, 1.0)

        # Point sizes from speed+turbulence.
        size_base = 6.0 + 9.0 * speed + 4.5 * turbulence
        frag_size_scale = 1.0 - 0.294 * (fragmentation ** 1.05)
        sizes = np.clip(size_base * (0.62 + 1.15 * (1.0 - depth_keep)) * frag_size_scale, 3.0, 30.0)

        colors = self.fg_cmap(color_field)
        rgb = np.clip(np.power(colors[:, :3], 0.85) * 1.14, 0.0, 1.0)
        luminance = np.sum(rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=1, keepdims=True)
        saturation_boost = 1.25
        colors[:, :3] = np.clip(luminance + (rgb - luminance) * saturation_boost, 0.0, 1.0)
        frag_alpha_scale = 1.0 - 0.336 * (fragmentation ** 1.1)
        colors[:, 3] = np.clip((0.30 + 0.62 * (1.0 - depth_keep)) * frag_alpha_scale, 0.12, 0.96)

        return p, sizes, colors

    def _render_month(self, month_float: float, immediate: bool = False) -> None:
        month_float = float(np.clip(month_float, 0.0, self.n_months - 1))
        self.current_month_float = month_float

        month_idx = int(np.round(month_float))
        month_idx = int(np.clip(month_idx, 0, self.n_months - 1))
        self.month_ptr = month_idx

        p, sizes, colors = self._particle_state(month_float)

        self.scatter.set_offsets(p)
        self.scatter.set_sizes(sizes)
        self.scatter.set_facecolors(colors)

        # Keep theta cursor markers synced to current interpolated month and theta state.
        _z_cur, t_cur, _fg_cur, _idx_cur = self._state_at_month(month_float)
        if self.theta_markers:
            for i, marker in enumerate(self.theta_markers):
                marker.set_data([month_float], [float(np.clip(t_cur[i], 0.0, 1.0))])

        # Keep slider position in sync with autoplay/keyboard-driven animation.
        if self.slider is not None and abs(float(self.slider.val) - month_float) > 1e-6:
            self._syncing_slider = True
            self.slider.set_val(month_float)
            self._syncing_slider = False

        if self.hud_text is not None:
            progress_ratio = 0.0 if self.n_months <= 1 else (self.current_month_float / (self.n_months - 1))
            elapsed = self.play_elapsed_seconds
            remaining = max(0.0, self.predicted_total_seconds - elapsed)
            ts_idx = int(self.month_to_index[self.month_ptr])
            current_date = datetime.fromtimestamp(int(self.data.timestamps[ts_idx]), tz=timezone.utc).strftime("%Y-%m-%d")
            self.hud_text.set_text(
                f"elapsed   : {elapsed:6.1f}s\n"
                f"pred_total: {self.predicted_total_seconds:6.1f}s\n"
                f"remaining : {remaining:6.1f}s\n"
                f"month     : {self.month_ptr+1:3d}/{self.n_months:3d}\n"
                f"date      : {current_date}\n"
                f"progress  : {progress_ratio*100:5.1f}%"
            )

        if immediate:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()

    def _start_transition(self, target_day: int) -> None:
        # Kept for compatibility with keyboard handlers; now directly seeks.
        target_day = int(np.clip(target_day, 0, self.n_months - 1))
        self._set_keep_mask_for_month(float(target_day))
        self._render_month(float(target_day))

    def _on_slider(self, val: float) -> None:
        if self._syncing_slider:
            return
        self._set_keep_mask_for_month(float(val))
        self._render_month(float(val))

    def _on_play_pause(self, _event) -> None:
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.last_timer_ts = time.perf_counter()
            self.timer.start()
        else:
            self.timer.stop()

    def _on_timer(self) -> None:
        now = time.perf_counter()

        if self.last_timer_ts is None:
            self.last_timer_ts = now

        dt = float(max(0.0, min(0.08, now - self.last_timer_ts)))
        self.last_timer_ts = now

        _z, t, _fg, _idx = self._state_at_month(self.current_month_float)
        dynamic_spin = 0.55 + 0.85 * float(t[4]) + 0.35 * float(t[2])
        # Always-positively advancing phase to avoid rotation direction flips.
        self.rotation_phase += dt * max(0.0288, (0.1008 + 0.1728 * dynamic_spin))

        if not self.playing:
            self._render_month(self.current_month_float)
            return

        self.play_elapsed_seconds += dt

        nxt_day = self.current_month_float + dt * self.months_per_second
        if nxt_day >= (self.n_months - 1):
            self.playing = False
            self.play_button.label.set_text("Play")
            self.timer.stop()
            self.play_elapsed_seconds = self.predicted_total_seconds
            self._render_month(float(self.n_months - 1))
            return

        self._set_keep_mask_for_month(nxt_day)
        self._render_month(nxt_day)

    def _on_key_press(self, event) -> None:
        if event.key == "right":
            self._start_transition(min(self.n_months - 1, int(round(self.current_month_float)) + 1))
        elif event.key == "left":
            self._start_transition(max(0, int(round(self.current_month_float)) - 1))

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
