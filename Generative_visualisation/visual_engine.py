"""
Simple real-time visual engine driven by generative parameter controls.

Expected parameter keys (0..1):
- motion_intensity
- particle_density
- distortion_strength
- noise_scale
- color_dynamics
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class VisualEngine:
    def __init__(self, width: int = 8, height: int = 8, base_particles: int = 120):
        self.width = width
        self.height = height
        self.base_particles = base_particles

        self.fig = None
        self.ax = None
        self.scatter = None
        self.info_text = None
        self.interpret_text = None
        self.heatmap_ax = None
        self.heatmap_text = None
        self.anim_timer = None

        self.positions = None
        self.velocities = None
        self.colors = None

        self.frame_idx = 0
        self.smoothing_alpha = 0.22
        self.tick_interval_ms = 20
        self.current_params = {
            "motion_intensity": 0.5,
            "particle_density": 0.5,
            "distortion_strength": 0.5,
            "noise_scale": 0.5,
            "color_dynamics": 0.5,
        }
        self.target_params = dict(self.current_params)

    def _ensure_plot(self):
        if self.fig is not None and self.ax is not None:
            return

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(self.width, self.height))
        self.fig.subplots_adjust(left=0.05, right=0.66, top=0.93, bottom=0.06)
        self.fig.patch.set_facecolor("#05060a")
        self.ax.set_facecolor("#05060a")
        self.ax.set_xlim(-1.25, 1.25)
        self.ax.set_ylim(-1.25, 1.25)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Live BTC Generative Visual", color="white", fontsize=12)

        self.info_text = self.fig.text(
            0.69,
            0.92,
            "",
            color="white",
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
            bbox={"facecolor": "#111827", "alpha": 0.55, "edgecolor": "#334155", "pad": 6},
        )

        self.interpret_text = self.fig.text(
            0.69,
            0.48,
            self._build_interpretation_panel(),
            color="white",
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
            bbox={"facecolor": "#0f172a", "alpha": 0.52, "edgecolor": "#334155", "pad": 6},
        )

        self.heatmap_ax = self.fig.add_axes([0.69, 0.09, 0.27, 0.05])
        gradient = np.linspace(0, 1, 256, dtype=np.float32).reshape(1, -1)
        self.heatmap_ax.imshow(gradient, aspect="auto", cmap="plasma", extent=[0, 1, 0, 1])
        self.heatmap_ax.set_yticks([])
        self.heatmap_ax.set_xticks([0.0, 0.5, 1.0])
        self.heatmap_ax.set_xticklabels(["cool", "mid", "warm"], color="white", fontsize=8)
        self.heatmap_ax.set_title("Particle colour heatmap (plasma)", color="white", fontsize=8, pad=2)
        self.heatmap_ax.set_facecolor("#0b1220")
        for spine in self.heatmap_ax.spines.values():
            spine.set_edgecolor("#334155")

        self.heatmap_text = self.fig.text(
            0.69,
            0.058,
            "Colour meaning: cooler tones = calmer/steady states, warmer tones = stronger activity and energy",
            color="#cbd5e1",
            fontsize=7.5,
            va="top",
            ha="left",
            family="sans-serif",
        )

        self.anim_timer = self.fig.canvas.new_timer(interval=self.tick_interval_ms)
        self.anim_timer.add_callback(self._tick)
        self.anim_timer.start()

    def _build_interpretation_panel(self) -> str:
        return (
            "Market interpretation\n"
            "- Higher volatility -> larger latent shifts\n"
            "  -> motion + distortion typically increase\n"
            "- Sudden regime changes -> stronger noise/jitter\n"
            "- Stable periods -> smoother flow + slower drift\n"
            "- PCA cluster drift reflects regime migration\n"
            "  (trend, consolidation, shock transitions)\n"
            "- Color dynamics rises when latent trajectory\n"
            "  changes direction/speed more frequently"
        )

    def _build_info_panel(self, params: dict[str, float]) -> str:
        motion = float(np.clip(params.get("motion_intensity", 0.5), 0.0, 1.0))
        density = float(np.clip(params.get("particle_density", 0.5), 0.0, 1.0))
        distortion = float(np.clip(params.get("distortion_strength", 0.5), 0.0, 1.0))
        noise = float(np.clip(params.get("noise_scale", 0.5), 0.0, 1.0))
        color_dyn = float(np.clip(params.get("color_dynamics", 0.5), 0.0, 1.0))

        return (
            "Parameter meaning\n"
            f"speed/motion : {motion:0.3f}  -> particle velocity\n"
            f"density       : {density:0.3f}  -> number of particles\n"
            f"distortion    : {distortion:0.3f}  -> swirl/curve force\n"
            f"noise         : {noise:0.3f}  -> random jitter\n"
            f"color dynamics: {color_dyn:0.3f}  -> color shift rate\n"
            "color map      : dark->warm plasma gradient"
        )

    def _initialize_particles(self, n_particles: int):
        angle = np.random.uniform(0, 2 * np.pi, size=n_particles)
        radius = np.sqrt(np.random.uniform(0.0, 1.0, size=n_particles))
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        self.positions = np.column_stack([x, y]).astype(np.float32)

        vx = np.random.normal(0.0, 0.01, size=n_particles)
        vy = np.random.normal(0.0, 0.01, size=n_particles)
        self.velocities = np.column_stack([vx, vy]).astype(np.float32)

        self.colors = cm.twilight(np.linspace(0, 1, n_particles))

    def _resize_particles(self, n_particles: int):
        current = 0 if self.positions is None else self.positions.shape[0]
        if current == n_particles:
            return

        if current == 0:
            self._initialize_particles(n_particles)
            return

        if n_particles < current:
            keep_idx = np.random.choice(current, size=n_particles, replace=False)
            self.positions = self.positions[keep_idx]
            self.velocities = self.velocities[keep_idx]
            self.colors = self.colors[keep_idx]
            return

        add = n_particles - current
        angle = np.random.uniform(0, 2 * np.pi, size=add)
        radius = np.sqrt(np.random.uniform(0.0, 1.0, size=add))
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        new_pos = np.column_stack([x, y]).astype(np.float32)

        vx = np.random.normal(0.0, 0.01, size=add)
        vy = np.random.normal(0.0, 0.01, size=add)
        new_vel = np.column_stack([vx, vy]).astype(np.float32)

        new_colors = cm.twilight(np.random.uniform(0.0, 1.0, size=add))

        self.positions = np.vstack([self.positions, new_pos])
        self.velocities = np.vstack([self.velocities, new_vel])
        self.colors = np.vstack([self.colors, new_colors])

    def render(self, params: dict[str, float]):
        self._ensure_plot()

        self.target_params = {
            "motion_intensity": float(np.clip(params.get("motion_intensity", 0.5), 0.0, 1.0)),
            "particle_density": float(np.clip(params.get("particle_density", 0.5), 0.0, 1.0)),
            "distortion_strength": float(np.clip(params.get("distortion_strength", 0.5), 0.0, 1.0)),
            "noise_scale": float(np.clip(params.get("noise_scale", 0.5), 0.0, 1.0)),
            "color_dynamics": float(np.clip(params.get("color_dynamics", 0.5), 0.0, 1.0)),
        }

        self._tick()

    def _tick(self):
        if self.fig is None or self.ax is None:
            return

        for key, target_value in self.target_params.items():
            self.current_params[key] = (
                (1.0 - self.smoothing_alpha) * self.current_params[key]
                + self.smoothing_alpha * target_value
            )

        motion = self.current_params["motion_intensity"]
        density = self.current_params["particle_density"]
        distortion = self.current_params["distortion_strength"]
        noise = self.current_params["noise_scale"]
        color_dyn = self.current_params["color_dynamics"]

        n_particles = int(self.base_particles + density * 520)
        self._resize_particles(n_particles)

        speed = 0.001 + motion * 0.008
        drift_angle = 0.10 * self.frame_idx
        drift = np.array([np.cos(drift_angle), np.sin(drift_angle)], dtype=np.float32)

        center = self.positions.copy()
        swirl = np.column_stack([-center[:, 1], center[:, 0]])
        swirl_strength = 0.0008 + distortion * 0.011

        noise_amp = 0.0006 + noise * 0.008
        noise_term = np.random.normal(0.0, noise_amp, size=self.positions.shape)

        self.velocities = 0.87 * self.velocities + speed * drift + swirl_strength * swirl + noise_term
        self.positions = self.positions + self.velocities

        radius = np.linalg.norm(self.positions, axis=1)
        out = radius > 1.25
        if np.any(out):
            self.positions[out] *= 0.15
            self.velocities[out] *= -0.35

        phase = (np.linspace(0, 1, n_particles) + self.frame_idx * (0.001 + 0.018 * color_dyn)) % 1.0
        self.colors = cm.plasma(phase)

        sizes = 8.0 + 26.0 * density + 8.0 * motion
        alpha = 0.22 + 0.62 * (0.55 * density + 0.45 * color_dyn)
        self.colors[:, 3] = np.clip(alpha, 0.15, 0.95)

        if self.scatter is None:
            self.scatter = self.ax.scatter(
                self.positions[:, 0],
                self.positions[:, 1],
                s=sizes,
                c=self.colors,
                edgecolors="none",
            )
        else:
            self.scatter.set_offsets(self.positions)
            self.scatter.set_sizes(np.full(n_particles, sizes))
            self.scatter.set_facecolors(self.colors)

        if self.info_text is not None:
            self.info_text.set_text(self._build_info_panel(self.current_params))

        self.frame_idx += 1
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
