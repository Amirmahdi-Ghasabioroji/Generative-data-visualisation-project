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
import matplotlib.patches as mpatches
from matplotlib import cm
from typing import Optional


class VisualEngine:
    def __init__(self, width: int = 8, height: int = 8, base_particles: int = 120):
        self.width = width
        self.height = height
        self.base_particles = 68

        self.fig = None
        self.ax = None
        self.scatter = None
        self.info_text = None
        self.regime_text = None
        self.market_text = None
        self.interpret_text = None
        self.heatmap_ax = None
        self.heatmap_text = None
        self.anim_timer = None

        self.positions = None
        self.velocities = None
        self.colors = None
        self.phase_offsets = None   # per-particle fixed colour phase seed

        self.starfield_scatter = None  # static background star field
        self.glow_patch = None         # soft central glow

        self.frame_idx = 0
        self.smoothing_alpha = 0.17
        self.tick_interval_ms = 20
        self.current_params = {
            "motion_intensity": 0.5,
            "particle_density": 0.5,
            "distortion_strength": 0.5,
            "noise_scale": 0.5,
            "color_dynamics": 0.5,
        }
        self.latest_model_params = dict(self.current_params)
        self.target_params = dict(self.current_params)
        self.arm_count = 3
        self.arm_twist = 5.8
        self.current_regime_id: Optional[int] = None
        self.current_regime_confidence = 0.0
        self.current_n_regimes = 0
        self.latest_regime_info: dict = {
            "regime_id": None,
            "confidence": 0.0,
            "n_regimes": 0,
        }
        self.latest_market_condition: dict = {
            "turbulence": 0.5,
            "trend_bias": 0.5,
            "distortion": 0.5,
            "fragmentation": 0.5,
            "velocity": 0.5,
            "quality": 0.0,
        }

        # Debounce: require N consecutive matching regime IDs before committing
        self._regime_debounce = 3
        self._pending_regime_id: Optional[int] = None
        self._pending_regime_count = 0

        # Subtle pulse on regime switch: briefly lifts motion + distortion only
        self._flash_frames = 0
        self._flash_max = 49  # ~0.8s at 24ms per frame

    def _ensure_plot(self):
        if self.fig is not None and self.ax is not None:
            return

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(self.width, self.height))
        self.fig.subplots_adjust(left=0.05, right=0.66, top=0.93, bottom=0.06)
        self.fig.patch.set_facecolor("#090c1f")
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

        self.regime_text = self.fig.text(
            0.69,
            0.66,
            self._build_regime_panel(),
            color="white",
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
            bbox={"facecolor": "#1f2937", "alpha": 0.55, "edgecolor": "#334155", "pad": 6},
        )

        self.market_text = self.fig.text(
            0.69,
            0.54,
            self._build_market_panel(),
            color="white",
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
            bbox={"facecolor": "#111827", "alpha": 0.56, "edgecolor": "#334155", "pad": 6},
        )

        self.interpret_text = self.fig.text(
            0.69,
            0.30,
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

        # Static background star field — never moves, creates depth
        rng = np.random.default_rng(42)
        star_x = rng.uniform(-1.25, 1.25, 90).astype(np.float32)
        star_y = rng.uniform(-1.25, 1.25, 90).astype(np.float32)
        star_s = rng.uniform(1.2, 5.5, 90).astype(np.float32)
        star_a = rng.uniform(0.18, 0.45, 90).astype(np.float32)
        star_colors = np.ones((90, 4), dtype=np.float32)
        star_colors[:, 3] = star_a
        self.starfield_scatter = self.ax.scatter(
            star_x, star_y, s=star_s, c=star_colors, edgecolors="none", zorder=1
        )

        # Soft central glow using a radial gradient via stacked transparent circles
        for r, a in [(0.20, 0.03), (0.12, 0.05), (0.06, 0.07)]:
            glow = mpatches.Circle((0, 0), r, color="#e8c87a", alpha=a, zorder=3, linewidth=0)
            self.ax.add_patch(glow)

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

    def _build_regime_panel(self) -> str:
        regime_id = self.latest_regime_info.get("regime_id")
        confidence = float(self.latest_regime_info.get("confidence", 0.0))
        n_regimes = int(max(1, self.latest_regime_info.get("n_regimes", 0)))

        regime_text = "R?/?: warming"
        if regime_id is not None and n_regimes > 0:
            regime_text = f"R{int(regime_id)}/{n_regimes} conf={confidence:0.3f}"

        return (
            "Regime information\n"
            f"{regime_text}"
        )

    def _build_market_panel(self) -> str:
        mc = self.latest_market_condition
        turbulence = float(np.clip(float(mc.get("turbulence", 0.5)), 0.0, 1.0))
        trend_bias = float(np.clip(float(mc.get("trend_bias", 0.5)), 0.0, 1.0))
        distortion = float(np.clip(float(mc.get("distortion", 0.5)), 0.0, 1.0))
        fragmentation = float(np.clip(float(mc.get("fragmentation", 0.5)), 0.0, 1.0))
        velocity = float(np.clip(float(mc.get("velocity", 0.5)), 0.0, 1.0))
        quality = float(np.clip(float(mc.get("quality", 0.0)), 0.0, 1.0))

        return (
            "Market condition\n"
            f"turbulence  : {turbulence:0.3f}\n"
            f"trend bias  : {trend_bias:0.3f}\n"
            f"distortion  : {distortion:0.3f}\n"
            f"fragmentation: {fragmentation:0.3f}\n"
            f"velocity    : {velocity:0.3f}\n"
            f"quality     : {quality:0.3f}"
        )

    def _build_info_panel(self, params: dict[str, float]) -> str:
        motion = float(params.get("motion_intensity", 0.5))
        density = float(params.get("particle_density", 0.5))
        distortion = float(params.get("distortion_strength", 0.5))
        noise = float(params.get("noise_scale", 0.5))
        color_dyn = float(params.get("color_dynamics", 0.5))

        return (
            "Parameter meaning\n"
            f"speed/motion : {motion:0.3f}\n"
            f"density       : {density:0.3f}\n"
            f"distortion    : {distortion:0.3f}\n"
            f"noise         : {noise:0.3f}\n"
            f"color dynamics: {color_dyn:0.3f}\n"
            "color map      : dark->warm plasma gradient"
        )

    def _apply_regime_style(self, regime_info: dict | None):
        if not regime_info:
            return

        regime_id = regime_info.get("regime_id")
        confidence = float(np.clip(float(regime_info.get("confidence", 0.0)), 0.0, 1.0))
        n_regimes = int(max(2, regime_info.get("n_regimes", 3)))

        if regime_id is None:
            self.current_regime_id = None
            self.current_regime_confidence = confidence
            self.current_n_regimes = n_regimes
            self._pending_regime_id = None
            self._pending_regime_count = 0
            return

        regime_id = int(np.clip(int(regime_id), 0, n_regimes - 1))

        # Debounce: only commit once the same regime appears N times in a row
        if regime_id == self._pending_regime_id:
            self._pending_regime_count += 1
        else:
            self._pending_regime_id = regime_id
            self._pending_regime_count = 1

        if self._pending_regime_count < self._regime_debounce:
            self.current_regime_confidence = confidence
            self.current_n_regimes = n_regimes
            return

        # Regime confirmed — trigger a subtle pulse if it actually changed
        if regime_id != self.current_regime_id:
            self._flash_frames = self._flash_max

        target_arm_count = int(np.clip(2 + regime_id, 2, 6))
        target_arm_twist = float(3.4 + 0.85 * regime_id)

        blend = 0.12 + 0.28 * confidence
        blended_count = (1.0 - blend) * float(self.arm_count) + blend * float(target_arm_count)
        self.arm_count = int(np.clip(int(round(blended_count)), 2, 6))
        self.arm_twist = float((1.0 - blend) * self.arm_twist + blend * target_arm_twist)

        self.current_regime_id = regime_id
        self.current_regime_confidence = confidence
        self.current_n_regimes = n_regimes

    def _initialize_particles(self, n_particles: int):
        self.positions = self._sample_galaxy_positions(n_particles)

        radius = np.linalg.norm(self.positions, axis=1) + 1e-6
        tangential = np.column_stack([-self.positions[:, 1], self.positions[:, 0]]) / radius[:, None]
        speed = 0.0011 + 0.0038 * (1.0 - np.clip(radius / 1.25, 0.0, 1.0))
        vxvy = tangential * speed[:, None]
        jitter = np.random.normal(0.0, 0.0010, size=(n_particles, 2))
        self.velocities = (vxvy + jitter).astype(np.float32)

        # Each particle gets a fixed phase seed so colour ripples across the cloud
        self.phase_offsets = np.random.uniform(0.0, 1.0, size=n_particles).astype(np.float32)
        self.colors = cm.twilight(np.linspace(0, 1, n_particles))

    def _sample_galaxy_positions(self, n_particles: int) -> np.ndarray:
        if n_particles <= 0:
            return np.empty((0, 2), dtype=np.float32)

        arm_idx = np.random.randint(0, self.arm_count, size=n_particles)
        base_theta = (2 * np.pi / self.arm_count) * arm_idx

        radius = np.random.beta(1.1, 2.6, size=n_particles) * 1.20
        theta = base_theta + radius * self.arm_twist + np.random.normal(0.0, 0.09, size=n_particles)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        core_n = min(n_particles, max(6, int(0.12 * n_particles)))
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
            if self.phase_offsets is not None:
                self.phase_offsets = self.phase_offsets[keep_idx]
            return

        add = n_particles - current
        new_pos = self._sample_galaxy_positions(add)

        radius = np.linalg.norm(new_pos, axis=1) + 1e-6
        tangential = np.column_stack([-new_pos[:, 1], new_pos[:, 0]]) / radius[:, None]
        speed = 0.0011 + 0.0038 * (1.0 - np.clip(radius / 1.25, 0.0, 1.0))
        new_vel = (tangential * speed[:, None] + np.random.normal(0.0, 0.0010, size=(add, 2))).astype(np.float32)

        new_colors = cm.twilight(np.random.uniform(0.0, 1.0, size=add))
        new_offsets = np.random.uniform(0.0, 1.0, size=add).astype(np.float32)

        self.positions = np.vstack([self.positions, new_pos])
        self.velocities = np.vstack([self.velocities, new_vel])
        self.colors = np.vstack([self.colors, new_colors])
        if self.phase_offsets is not None:
            self.phase_offsets = np.concatenate([self.phase_offsets, new_offsets])
        else:
            self.phase_offsets = new_offsets

    def render(
        self,
        params: dict[str, float],
        regime_info: dict | None = None,
        market_condition: dict | None = None,
    ):
        self._ensure_plot()
        self.latest_model_params = {
            "motion_intensity": float(params.get("motion_intensity", 0.5)),
            "particle_density": float(params.get("particle_density", 0.5)),
            "distortion_strength": float(params.get("distortion_strength", 0.5)),
            "noise_scale": float(params.get("noise_scale", 0.5)),
            "color_dynamics": float(params.get("color_dynamics", 0.5)),
        }
        if regime_info is not None:
            self.latest_regime_info = {
                "regime_id": regime_info.get("regime_id"),
                "confidence": float(regime_info.get("confidence", 0.0)),
                "n_regimes": int(regime_info.get("n_regimes", 0)),
            }
        if market_condition is not None:
            self.latest_market_condition = {
                "turbulence": float(market_condition.get("turbulence", 0.5)),
                "trend_bias": float(market_condition.get("trend_bias", 0.5)),
                "distortion": float(market_condition.get("distortion", 0.5)),
                "fragmentation": float(market_condition.get("fragmentation", 0.5)),
                "velocity": float(market_condition.get("velocity", 0.5)),
                "quality": float(market_condition.get("quality", 0.0)),
            }
        self._apply_regime_style(self.latest_regime_info)

        self.target_params = {
            "motion_intensity": float(np.clip(params.get("motion_intensity", 0.5), 0.0, 1.0)),
            "particle_density": float(np.clip(params.get("particle_density", 0.5), 0.0, 1.0)),
            "distortion_strength": float(np.clip(params.get("distortion_strength", 0.5), 0.0, 1.0)),
            "noise_scale": float(np.clip(params.get("noise_scale", 0.5), 0.0, 1.0)),
            "color_dynamics": float(np.clip(params.get("color_dynamics", 0.5), 0.0, 1.0)),
        }

        if self.info_text is not None:
            self.info_text.set_text(self._build_info_panel(self.latest_model_params))
        if self.regime_text is not None:
            self.regime_text.set_text(self._build_regime_panel())
        if self.market_text is not None:
            self.market_text.set_text(self._build_market_panel())

        if self.fig is not None:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        

    def _tick(self):
        if self.fig is None or self.ax is None:
            return

        self._apply_regime_style(self.latest_regime_info)

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

        n_particles = int(self.base_particles + density * 280)
        self._resize_particles(n_particles)

        center = self.positions.copy()
        radius = np.linalg.norm(center, axis=1) + 1e-6
        theta = np.arctan2(center[:, 1], center[:, 0])

        # Pulse on regime switch: boost motion, distortion, particle size and brightness
        flash_t = 0.0
        if self._flash_frames > 0:
            flash_t = (self._flash_frames / self._flash_max) ** 0.5
            motion = float(np.clip(motion + 0.35 * flash_t, 0.0, 1.0))
            distortion = float(np.clip(distortion + 0.25 * flash_t, 0.0, 1.0))
            noise = float(np.clip(noise + 0.15 * flash_t, 0.0, 1.0))
            self._flash_frames -= 1

        # Spiral flow in polar coordinates gives a continuous galaxy-like curl.
        radial_norm = np.clip(radius / 1.25, 0.0, 1.0)
        omega = (0.010 + 0.052 * motion) * (1.25 - 0.55 * radial_norm)
        arm_phase = 0.35 * np.sin(self.arm_count * theta - 0.035 * self.frame_idx)
        theta_next = theta + omega + 0.020 * distortion * arm_phase

        inward = 0.00055 + 0.0028 * distortion
        breathing = 0.00035 * np.sin(0.028 * self.frame_idx + 4.0 * theta)
        radial_noise = np.random.normal(0.0, 0.00025 + 0.0009 * noise, size=radius.shape)
        radius_next = np.clip(radius - inward * radial_norm + breathing + radial_noise, 0.04, 1.22)

        next_pos = np.column_stack([radius_next * np.cos(theta_next), radius_next * np.sin(theta_next)]).astype(np.float32)
        self.velocities = 0.82 * self.velocities + 0.18 * (next_pos - self.positions)
        self.positions = next_pos

        radius = np.linalg.norm(self.positions, axis=1)
        out = radius > 1.25
        if np.any(out):
            self.positions[out] *= 0.22
            self.velocities[out] *= -0.25

        radius_now = np.linalg.norm(self.positions, axis=1)
        warm_bias = np.clip(1.0 - radius_now / 1.25, 0.0, 1.0)
        # Per-particle phase offset makes colour ripple across cloud rather than pulsing in unison
        offsets = self.phase_offsets if self.phase_offsets is not None else 0.0
        phase = (offsets + 0.75 * warm_bias + self.frame_idx * (0.0006 + 0.012 * color_dyn)) % 1.0
        self.colors = cm.magma(phase)

        # Strengthen colour intensity (vivid highlights while preserving palette ordering)
        self.colors[:, :3] = np.clip(self.colors[:, :3] ** 0.82, 0.0, 1.0)

        core_emphasis = np.clip(1.0 - radius_now / 1.25, 0.0, 1.0)

        # Per-particle size and brightness driven by velocity magnitude.
        # Fast particles (energised by volatile BTC moments) are larger and brighter;
        # slow drifting particles in calm regimes stay small and dim.
        vel_mag = np.linalg.norm(self.velocities, axis=1)
        expected_max = 0.0007 + motion * 0.0062 + 2e-4  # rough expected ceiling
        vel_norm = np.clip(vel_mag / (expected_max + 1e-8), 0.0, 1.0)

        base_size = 4.2 + 12.0 * density + 5.5 * motion + 18.0 * core_emphasis + 16.0 * flash_t
        sizes = base_size * (0.55 + 0.95 * vel_norm)  # per-particle: 0.55x–1.5x base

        alpha_base = 0.18 + 0.55 * (0.55 * density + 0.45 * color_dyn)
        per_alpha = alpha_base * (0.60 + 0.65 * core_emphasis) + 0.28 * vel_norm
        self.colors[:, 3] = np.clip(per_alpha + 0.25 * flash_t, 0.15, 1.0)

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
            self.scatter.set_sizes(sizes)
            self.scatter.set_facecolors(self.colors)

        if self.info_text is not None:
            self.info_text.set_text(self._build_info_panel(self.latest_model_params))
        if self.regime_text is not None:
            self.regime_text.set_text(self._build_regime_panel())
        if self.market_text is not None:
            self.market_text.set_text(self._build_market_panel())

        self.frame_idx += 1
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
