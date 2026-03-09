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
        self.tick_interval_ms = 24
        self.current_params = {
            "motion_intensity": 0.5,
            "particle_density": 0.5,
            "distortion_strength": 0.5,
            "noise_scale": 0.5,
            "color_dynamics": 0.5,
        }
        self.target_params = dict(self.current_params)
        self.arm_count = 3
        self.arm_twist = 4.4

    def _ensure_plot(self):
        if self.fig is not None and self.ax is not None:
            return

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(self.width, self.height))
        self.fig.subplots_adjust(left=0.05, right=0.66, top=0.93, bottom=0.06)
        self.fig.patch.set_facecolor("#03040a")
        self.ax.set_facecolor("#03040a")
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

    def _build_info_panel(self, params: dict[str, float], target_params: dict[str, float] | None = None) -> str:
        motion = float(np.clip(params.get("motion_intensity", 0.5), 0.0, 1.0))
        density = float(np.clip(params.get("particle_density", 0.5), 0.0, 1.0))
        distortion = float(np.clip(params.get("distortion_strength", 0.5), 0.0, 1.0))
        noise = float(np.clip(params.get("noise_scale", 0.5), 0.0, 1.0))
        color_dyn = float(np.clip(params.get("color_dynamics", 0.5), 0.0, 1.0))

        return (
            "Parameter meaning\n"
            f"speed/motion : {motion:0.4f} -> particle velocity\n"
            f"density       : {density:0.4f} -> number of particles\n"
            f"distortion    : {distortion:0.4f} -> swirl/curve force\n"
            f"noise         : {noise:0.4f} -> random jitter\n"
            f"color dynamics: {color_dyn:0.4f} -> color shift rate\n"
            "color map      : dark->warm plasma gradient"
        )

    def _initialize_particles(self, n_particles: int):
        self.positions = self._sample_galaxy_positions(n_particles)

        radius = np.linalg.norm(self.positions, axis=1) + 1e-6
        tangential = np.column_stack([-self.positions[:, 1], self.positions[:, 0]]) / radius[:, None]
        speed = 0.0012 + 0.0042 * (1.0 - np.clip(radius / 1.25, 0.0, 1.0))
        vxvy = tangential * speed[:, None]
        jitter = np.random.normal(0.0, 0.0016, size=(n_particles, 2))
        self.velocities = (vxvy + jitter).astype(np.float32)

        self.colors = cm.twilight(np.linspace(0, 1, n_particles))

    def _sample_galaxy_positions(self, n_particles: int) -> np.ndarray:
        if n_particles <= 0:
            return np.empty((0, 2), dtype=np.float32)

        arm_idx = np.random.randint(0, self.arm_count, size=n_particles)
        base_theta = (2 * np.pi / self.arm_count) * arm_idx

        radius = np.random.beta(1.2, 2.8, size=n_particles) * 1.18
        theta = base_theta + radius * self.arm_twist + np.random.normal(0.0, 0.14, size=n_particles)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        core_n = min(n_particles, max(8, int(0.18 * n_particles)))
        core_radius = np.random.beta(1.0, 6.0, size=core_n) * 0.22
        core_theta = np.random.uniform(0.0, 2 * np.pi, size=core_n)
        x[:core_n] = core_radius * np.cos(core_theta)
        y[:core_n] = core_radius * np.sin(core_theta)

        return np.column_stack([x, y]).astype(np.float32)

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
        new_pos = self._sample_galaxy_positions(add)

        radius = np.linalg.norm(new_pos, axis=1) + 1e-6
        tangential = np.column_stack([-new_pos[:, 1], new_pos[:, 0]]) / radius[:, None]
        speed = 0.0012 + 0.0042 * (1.0 - np.clip(radius / 1.25, 0.0, 1.0))
        new_vel = (tangential * speed[:, None] + np.random.normal(0.0, 0.0016, size=(add, 2))).astype(np.float32)

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

        if self.info_text is not None:
            self.info_text.set_text(self._build_info_panel(self.current_params, self.target_params))

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

        speed = 0.0007 + motion * 0.0052
        drift_angle = 0.06 * self.frame_idx
        drift = np.array([np.cos(drift_angle), np.sin(drift_angle)], dtype=np.float32)

        center = self.positions.copy()
        radius = np.linalg.norm(center, axis=1) + 1e-6
        tangential = np.column_stack([-center[:, 1], center[:, 0]]) / radius[:, None]
        radial_inward = -center / radius[:, None]

        swirl_strength = 0.0011 + distortion * 0.007
        core_pull_strength = 0.0015 + 0.0048 * (1.0 - np.clip(radius / 1.25, 0.0, 1.0))

        noise_amp = 0.00045 + noise * 0.0055
        noise_term = np.random.normal(0.0, noise_amp, size=self.positions.shape)

        self.velocities = (
            0.93 * self.velocities
            + speed * drift
            + swirl_strength * tangential
            + core_pull_strength[:, None] * radial_inward
            + noise_term
        )
        self.positions = self.positions + self.velocities

        radius = np.linalg.norm(self.positions, axis=1)
        out = radius > 1.25
        if np.any(out):
            self.positions[out] *= 0.15
            self.velocities[out] *= -0.35

        radius_now = np.linalg.norm(self.positions, axis=1)
        warm_bias = np.clip(1.0 - radius_now / 1.25, 0.0, 1.0)
        phase = (0.25 * np.linspace(0, 1, n_particles) + 0.75 * warm_bias + self.frame_idx * (0.0006 + 0.012 * color_dyn)) % 1.0
        self.colors = cm.magma(phase)

        # Strengthen colour intensity (vivid highlights while preserving palette ordering)
        self.colors[:, :3] = np.clip(self.colors[:, :3] ** 0.82, 0.0, 1.0)

        core_emphasis = np.clip(1.0 - radius_now / 1.25, 0.0, 1.0)
        sizes = 6.0 + 18.0 * density + 6.0 * motion + 20.0 * core_emphasis
        alpha = 0.22 + 0.62 * (0.55 * density + 0.45 * color_dyn)
        self.colors[:, 3] = np.clip(alpha * (0.72 + 0.55 * core_emphasis), 0.20, 1.0)

        if self.scatter is None:
            self.scatter = self.ax.scatter(
                self.positions[:, 0],
                self.positions[:, 1],
                s=sizes,
                c=self.colors,
                edgecolors="none",
                zorder=2,
            )
        else:
            self.scatter.set_offsets(self.positions)
            self.scatter.set_sizes(np.full(n_particles, sizes))
            self.scatter.set_facecolors(self.colors)

        if self.info_text is not None:
            self.info_text.set_text(self._build_info_panel(self.current_params, self.target_params))

        self.frame_idx += 1
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
