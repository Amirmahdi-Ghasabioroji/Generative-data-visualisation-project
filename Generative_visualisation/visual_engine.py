"""
Simple real-time visual engine driven by generative parameter controls.

Expected parameter keys (0..1), aligned with historical theta semantics:
- turbulence
- trend_bias
- distortion
- fragmentation
- velocity

Legacy live keys are still accepted for compatibility:
- motion_intensity -> turbulence
- color_dynamics -> trend_bias
- distortion_strength -> distortion
- noise_scale -> fragmentation
- particle_density -> velocity
"""

from __future__ import annotations

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib import cm
from typing import Optional


class VisualEngine:
    def __init__(self, width: int = 8, height: int = 8, base_particles: int = 120):
        self.width = width
        self.height = height
        self.base_particles = 185

        self.fig = None
        self.ax = None
        self.scatter = None
        self.tail_collection = None
        self.info_text = None
        self.social_text = None
        self.regime_text = None
        self.market_text = None
        self.interpret_text = None
        self.heatmap_ax = None
        self.heatmap_text = None
        self.anim_timer = None

        self.positions = None
        self.trail_history = deque(maxlen=5)
        self.velocities = None
        self.colors = None
        self.phase_offsets = None   # per-particle fixed colour phase seed

        self.starfield_scatter = None  # static background star field
        self.glow_patch = None         # soft central glow

        self.frame_idx = 0
        # Lower alpha smooths parameter transitions between sparse upstream updates.
        self.smoothing_alpha = 0.055
        # Global temporal scale for live animation. 0.90 => ~10% slower overall.
        self.global_speed_scale = 0.90
        self.tick_interval_ms = 16
        self.current_params = {
            "turbulence": 0.5,
            "trend_bias": 0.5,
            "distortion": 0.5,
            "fragmentation": 0.5,
            "velocity": 0.5,
        }
        self.latest_model_params = dict(self.current_params)
        self.target_params = dict(self.current_params)
        self.arm_count = 3
        self.arm_twist = 5.8
        self.current_regime_id: Optional[int] = None
        self.current_regime_confidence = 0.0
        self.current_n_regimes = 0
        self.tail_base_alpha = 0.065
        # Particle count controls with hysteresis to avoid rapid resize thrashing.
        self._density_ema = 0.50
        self._resize_hysteresis = 14
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
        self._flash_max = 58  # longer, softer cinematic pulse

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

        self.social_text = self.fig.text(
            0.69,
            0.75,
            self._build_social_panel(),
            color="white",
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
            bbox={"facecolor": "#0f172a", "alpha": 0.55, "edgecolor": "#334155", "pad": 6},
        )

        self.regime_text = self.fig.text(
            0.69,
            0.60,
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
            0.53,
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
            0.36,
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
        self.heatmap_ax.imshow(gradient, aspect="auto", cmap="RdYlGn", extent=[0, 1, 0, 1])
        self.heatmap_ax.set_yticks([])
        self.heatmap_ax.set_xticks([0.0, 0.5, 1.0])
        self.heatmap_ax.set_xticklabels(["bearish (red)", "neutral", "bullish (green)"], color="white", fontsize=8)
        self.heatmap_ax.set_title("Particle colour heatmap (Classic R/G)", color="white", fontsize=8, pad=2)
        self.heatmap_ax.set_facecolor("#0b1220")
        for spine in self.heatmap_ax.spines.values():
            spine.set_edgecolor("#334155")

        self.heatmap_text = self.fig.text(
            0.69,
            0.17,
            "Colour meaning: red=bearish, yellow=neutral, green=bullish",
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
            "Market interpretation (what the market looks like)\n"
            "- Higher volatility -> larger latent shifts\n"
            "  -> turbulence + distortion typically increase\n"
            "- Sudden regime changes -> stronger noise/jitter\n"
            "- Stable periods -> smoother flow + slower drift\n"
            "- Velocity/speed shows how fast the latent state\n"
            "  is changing right now (tempo of regime movement)\n"
            "- PCA cluster drift reflects regime migration\n"
            "  (trend, consolidation, shock transitions)\n"
            "- Trend bias drives bearish<->bullish color drift"
        )

    def _build_regime_panel(self) -> str:
        regime_id = self.latest_regime_info.get("regime_id")
        confidence = float(self.latest_regime_info.get("confidence", 0.0))
        n_regimes = int(max(1, self.latest_regime_info.get("n_regimes", 0)))

        regime_text = "R?/?: warming"
        if regime_id is not None and n_regimes > 0:
            regime_text = f"R{int(regime_id)}/{n_regimes} confidence={confidence:0.3f}"

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
            "─────────────────\n"
            f"turbulence  : {turbulence:0.3f}  [volatility]\n"
            f"trend bias  : {trend_bias:0.3f}  [0=bear 1=bull]\n"
            f"distortion  : {distortion:0.3f}  [regime shift]\n"
            f"fragmentation: {fragmentation:0.3f}  [choppiness]\n"
            f"velocity    : {velocity:0.3f}  [state-change speed]\n"
            f"quality     : {quality:0.3f}  [confidence]"
        )

    def _build_social_panel(self) -> str:
        mc = self.latest_market_condition

        social_blend_weight = float(np.clip(float(mc.get("social_blend_weight", 0.0)), 0.0, 1.0))
        social_quality = float(np.clip(float(mc.get("social_quality", 0.0)), 0.0, 1.0))
        social_posts = int(max(0, round(float(mc.get("social_posts", 0.0)))))
        social_age_sec = float(max(0.0, float(mc.get("social_age_sec", 9999.0))))
        social_stale = float(mc.get("social_stale", 1.0)) >= 0.5
        social_valid = float(mc.get("social_valid", 0.0)) >= 0.5
        social_error_streak = int(max(0, round(float(mc.get("social_error_streak", 0.0)))))

        if social_valid and not social_stale:
            social_status = "OK"
        elif social_stale:
            social_status = "STALE"
        else:
            social_status = "LOW_CONF"

        return (
            "Social validation\n"
            "─────────────────\n"
            f"status      : {social_status}\n"
            f"blend weight: {social_blend_weight:0.3f} [social influence, scale with quality]\n"
            f"social quality: {social_quality:0.3f} [confidence score, relevancy]\n"
            f"posts  : {social_posts:d} [posts in rolling window]\n"
            f"age(sec)    : {social_age_sec:0.1f}\n"
            f"err streak  : {social_error_streak:d}"
        )

    @staticmethod
    def _normalize_live_params(params: dict[str, float]) -> dict[str, float]:
        # Accept both canonical semantic names and legacy renderer names.
        return {
            "turbulence": float(np.clip(params.get("turbulence", params.get("motion_intensity", 0.5)), 0.0, 1.0)),
            "trend_bias": float(np.clip(params.get("trend_bias", params.get("color_dynamics", 0.5)), 0.0, 1.0)),
            "distortion": float(np.clip(params.get("distortion", params.get("distortion_strength", 0.5)), 0.0, 1.0)),
            "fragmentation": float(np.clip(params.get("fragmentation", params.get("noise_scale", 0.5)), 0.0, 1.0)),
            "velocity": float(np.clip(params.get("velocity", params.get("particle_density", 0.5)), 0.0, 1.0)),
        }

    def _build_info_panel(self, params: dict[str, float]) -> str:
        turbulence = float(params.get("turbulence", 0.5))
        trend_bias = float(params.get("trend_bias", 0.5))
        distortion = float(params.get("distortion", 0.5))
        fragmentation = float(params.get("fragmentation", 0.5))
        velocity = float(params.get("velocity", 0.5))

        return (
            "Live theta-aligned parameters\n"
            f"turbulence   : {turbulence:0.3f}\n"
            f"trend bias   : {trend_bias:0.3f}\n"
            f"distortion   : {distortion:0.3f}\n"
            f"fragmentation: {fragmentation:0.3f}\n"
            f"velocity     : {velocity:0.3f}\n"
            "color map      : Red->Green (bear->bull heatmap)\n"
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
        self.colors = cm.RdYlGn(np.linspace(0, 1, n_particles))
        self.trail_history.clear()
        self.trail_history.append(self.positions.copy())

    def _sample_galaxy_positions(self, n_particles: int) -> np.ndarray:
        if n_particles <= 0:
            return np.empty((0, 2), dtype=np.float32)

        # 40% spiral arms + 60% uniform disk fill for broader coverage
        n_spiral = int(0.40 * n_particles)
        n_disk = n_particles - n_spiral

        # Spiral arm particles
        if n_spiral > 0:
            arm_idx = np.random.randint(0, self.arm_count, size=n_spiral)
            base_theta_spiral = (2 * np.pi / self.arm_count) * arm_idx
            radius_spiral = np.random.beta(1.1, 2.6, size=n_spiral) * 1.20
            theta_spiral = base_theta_spiral + radius_spiral * self.arm_twist + np.random.normal(0.0, 0.09, size=n_spiral)
            x_spiral = radius_spiral * np.cos(theta_spiral)
            y_spiral = radius_spiral * np.sin(theta_spiral)

        # Uniform disk particles
        if n_disk > 0:
            theta_disk = np.random.uniform(0.0, 2 * np.pi, size=n_disk)
            radius_disk = np.sqrt(np.random.uniform(0.0, 1.0, size=n_disk)) * 1.20
            x_disk = radius_disk * np.cos(theta_disk)
            y_disk = radius_disk * np.sin(theta_disk)

        # Combine
        if n_spiral > 0 and n_disk > 0:
            x = np.concatenate([x_spiral, x_disk])
            y = np.concatenate([y_spiral, y_disk])
        elif n_spiral > 0:
            x = x_spiral
            y = y_spiral
        else:
            x = x_disk
            y = y_disk

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
            if len(self.trail_history) > 0:
                new_history = deque(maxlen=self.trail_history.maxlen)
                for frame in self.trail_history:
                    if frame.shape[0] == current:
                        new_history.append(frame[keep_idx])
                self.trail_history = new_history
            return

        add = n_particles - current
        new_pos = self._sample_galaxy_positions(add)

        radius = np.linalg.norm(new_pos, axis=1) + 1e-6
        tangential = np.column_stack([-new_pos[:, 1], new_pos[:, 0]]) / radius[:, None]
        speed = 0.0011 + 0.0038 * (1.0 - np.clip(radius / 1.25, 0.0, 1.0))
        new_vel = (tangential * speed[:, None] + np.random.normal(0.0, 0.0010, size=(add, 2))).astype(np.float32)

        new_colors = cm.RdYlGn(np.random.uniform(0.0, 1.0, size=add))
        new_offsets = np.random.uniform(0.0, 1.0, size=add).astype(np.float32)

        self.positions = np.vstack([self.positions, new_pos])
        self.velocities = np.vstack([self.velocities, new_vel])
        self.colors = np.vstack([self.colors, new_colors])
        if self.phase_offsets is not None:
            self.phase_offsets = np.concatenate([self.phase_offsets, new_offsets])
        else:
            self.phase_offsets = new_offsets
        if len(self.trail_history) > 0:
            new_history = deque(maxlen=self.trail_history.maxlen)
            for frame in self.trail_history:
                if frame.shape[0] == current:
                    new_history.append(np.vstack([frame, new_pos]))
            self.trail_history = new_history

    def render(
        self,
        params: dict[str, float],
        regime_info: dict | None = None,
        market_condition: dict | None = None,
    ):
        self._ensure_plot()
        normalized_params = self._normalize_live_params(params)
        # Keep raw model values for HUD text; animation uses clipped target_params below.
        self.latest_model_params = dict(normalized_params)
        if regime_info is not None:
            self.latest_regime_info = {
                "regime_id": regime_info.get("regime_id"),
                "confidence": float(regime_info.get("confidence", 0.0)),
                "n_regimes": int(regime_info.get("n_regimes", 0)),
            }
        if market_condition is not None:
            current = dict(self.latest_market_condition)
            current.update(
                {
                    "turbulence": float(market_condition.get("turbulence", 0.5)),
                    "trend_bias": float(market_condition.get("trend_bias", 0.5)),
                    "distortion": float(market_condition.get("distortion", 0.5)),
                    "fragmentation": float(market_condition.get("fragmentation", 0.5)),
                    "velocity": float(market_condition.get("velocity", 0.5)),
                    "quality": float(market_condition.get("quality", 0.0)),
                    "social_blend_weight": float(market_condition.get("social_blend_weight", current.get("social_blend_weight", 0.0))),
                    "social_quality": float(market_condition.get("social_quality", current.get("social_quality", 0.0))),
                    "social_posts": float(market_condition.get("social_posts", current.get("social_posts", 0.0))),
                    "social_age_sec": float(market_condition.get("social_age_sec", current.get("social_age_sec", 9999.0))),
                    "social_stale": float(market_condition.get("social_stale", current.get("social_stale", 1.0))),
                    "social_valid": float(market_condition.get("social_valid", current.get("social_valid", 0.0))),
                    "social_error_streak": float(market_condition.get("social_error_streak", current.get("social_error_streak", 0.0))),
                }
            )
            self.latest_market_condition = current
        self._apply_regime_style(self.latest_regime_info)

        self.target_params = dict(normalized_params)

        if self.info_text is not None:
            self.info_text.set_text(self._build_info_panel(self.latest_model_params))
        if self.social_text is not None:
            self.social_text.set_text(self._build_social_panel())
        if self.regime_text is not None:
            self.regime_text.set_text(self._build_regime_panel())
        if self.market_text is not None:
            self.market_text.set_text(self._build_market_panel())

    def pump_events(self):
        """Process GUI events so timer-driven animation can run smoothly."""
        if self.fig is None:
            return
        plt.pause(0.001)

        

    def _tick(self):
        if self.fig is None or self.ax is None:
            return

        self._apply_regime_style(self.latest_regime_info)

        # Exponential smoothing avoids visual jumps when upstream updates are bursty.
        for key, target_value in self.target_params.items():
            self.current_params[key] = (
                (1.0 - self.smoothing_alpha) * self.current_params[key]
                + self.smoothing_alpha * target_value
            )

        turbulence = self.current_params["turbulence"]
        trend_bias = self.current_params["trend_bias"]
        distortion = self.current_params["distortion"]
        fragmentation = self.current_params["fragmentation"]
        velocity = self.current_params["velocity"]
        speed_scale = float(np.clip(self.global_speed_scale, 0.70, 1.20))

        # Derived renderer controls preserve existing visual behavior while
        # keeping externally visible parameters theta-aligned.
        motion = float(np.clip(0.52 * velocity + 0.48 * turbulence, 0.0, 1.0))
        density_raw = float(np.clip(1.0 - 0.72 * fragmentation, 0.0, 1.0))
        self._density_ema = float(0.90 * self._density_ema + 0.10 * density_raw)
        density = self._density_ema
        noise = float(np.clip(0.62 * fragmentation + 0.38 * turbulence, 0.0, 1.0))
        color_dyn = float(np.clip(0.56 * abs(trend_bias - 0.5) * 2.0 + 0.44 * velocity, 0.0, 1.0))

        # Density modulates particle budget while keeping a minimum visual baseline.
        desired_particles = int(self.base_particles + density * 440)
        current_particles = 0 if self.positions is None else int(self.positions.shape[0])
        if current_particles == 0 or abs(desired_particles - current_particles) >= self._resize_hysteresis:
            self._resize_particles(desired_particles)

        center = self.positions.copy()
        radius = np.linalg.norm(center, axis=1) + 1e-6
        theta = np.arctan2(center[:, 1], center[:, 0])

        # Pulse on regime switch: boost motion, distortion, particle size and brightness
        flash_t = 0.0
        if self._flash_frames > 0:
            flash_t = (self._flash_frames / self._flash_max) ** 0.5
            motion = float(np.clip(motion + 0.26 * flash_t, 0.0, 1.0))
            distortion = float(np.clip(distortion + 0.18 * flash_t, 0.0, 1.0))
            noise = float(np.clip(noise + 0.10 * flash_t, 0.0, 1.0))
            self._flash_frames -= 1

        # Spiral flow with loostened constraints for smoother drift
        radial_norm = np.clip(radius / 1.25, 0.0, 1.0)
        omega = speed_scale * (0.0096 + 0.0580 * motion) * (1.25 - 0.46 * radial_norm)
        arm_phase = 0.38 * np.sin(self.arm_count * theta - (0.040 * speed_scale) * self.frame_idx)
        theta_next = theta + omega + speed_scale * 0.0240 * distortion * arm_phase

        inward = speed_scale * (0.00039 + 0.0022 * distortion)
        breathing = 0.00040 * np.sin((0.022 * speed_scale) * self.frame_idx + 3.5 * theta)
        radial_noise = np.random.normal(0.0, 0.92 * (0.00035 + 0.0012 * noise), size=radius.shape)
        radius_next = np.clip(radius - inward * radial_norm + breathing + radial_noise, 0.04, 1.22)

        next_pos = np.column_stack([radius_next * np.cos(theta_next), radius_next * np.sin(theta_next)]).astype(np.float32)
        # Loosen constraint: reduce spiral snap-back, allow more natural drift
        self.velocities = 0.80 * self.velocities + 0.20 * (next_pos - self.positions)
        self.positions = next_pos

        radius = np.linalg.norm(self.positions, axis=1)
        out = radius > 1.25
        if np.any(out):
            self.positions[out] *= 0.22
            self.velocities[out] *= -0.25

        radius_now = np.linalg.norm(self.positions, axis=1)
        warm_bias = np.clip(1.0 - radius_now / 1.25, 0.0, 1.0)
        # Colour is anchored to live market trend_bias:
        # bearish(0) -> red, neutral(0.5) -> yellow, bullish(1) -> green.
        # Keep a light spatial/time texture so the cloud is not flat.
        offsets = self.phase_offsets if self.phase_offsets is not None else 0.0
        texture = 0.12 * (offsets - 0.5) + 0.10 * (warm_bias - 0.5)
        shimmer = 0.04 * np.sin(self.frame_idx * (speed_scale * (0.003 + 0.014 * color_dyn)) + 6.0 * offsets)
        phase = np.clip(trend_bias + texture + shimmer, 0.0, 1.0)
        self.colors = cm.RdYlGn(phase)

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

        # Build a short fading multi-segment trail history for more natural tails.
        self.trail_history.append(self.positions.copy())
        history = list(self.trail_history)
        can_draw_trails = (
            len(history) >= 2
            and self.positions is not None
            and self.positions.shape[0] > 0
            and all(h.shape == self.positions.shape for h in history)
        )

        if can_draw_trails:
            all_segments = []
            all_colors = []
            all_widths = []

            n_links = len(history) - 1
            for i in range(n_links):
                p0 = history[i]
                p1 = history[i + 1]
                seg = np.stack([p0, p1], axis=1)
                all_segments.append(seg)

                # Older segments are thinner/fainter; newer segments are slightly stronger.
                age_mix = float((i + 1) / max(1, n_links))
                c = self.colors.copy()
                c[:, 3] = np.clip(
                    (0.35 + 0.65 * age_mix) * (self.tail_base_alpha + 0.09 * vel_norm)
                    + 0.06 * flash_t,
                    0.03,
                    0.20,
                )
                all_colors.append(c)

                w = (0.20 + 0.55 * age_mix) * (0.45 + 0.80 * vel_norm)
                all_widths.append(w)

            segments = np.concatenate(all_segments, axis=0)
            colors = np.concatenate(all_colors, axis=0)
            widths = np.concatenate(all_widths, axis=0)

            if self.tail_collection is None:
                self.tail_collection = LineCollection(
                    segments,
                    colors=colors,
                    linewidths=widths,
                    capstyle="round",
                    zorder=1.7,
                )
                self.ax.add_collection(self.tail_collection)
            else:
                self.tail_collection.set_segments(segments)
                self.tail_collection.set_color(colors)
                self.tail_collection.set_linewidths(widths)
        elif self.tail_collection is not None:
            self.tail_collection.set_segments([])

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
        
