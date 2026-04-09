"""
Live BTC visual-parameter pipeline.

Connects the existing real-time Bitcoin pipeline + current PCA stage
with the unsupervised latent-to-visual mapper.

Flow:
Binance BTC stream -> Real_time_Crypto feature extraction -> PCA (3D)
-> latent_visual_mapper -> generative parameter dict
"""

from __future__ import annotations

import asyncio
from collections import deque
import os
from pathlib import Path
from typing import Deque, Dict, Optional
import sys
import time

import numpy as np
from binance import AsyncClient, BinanceSocketManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Data_Pipeline import Real_time_Crypto as rtc
from Data_Pipeline.Static_Bluesky import LiveSocialSentimentPoller
from Generative_visualisation import live_btc_pca_visual as pca_runner
from AI_systems.latent_visual_mapper import (
    StreamingLatentVisualMapper,
    set_default_mapper,
)
from Generative_visualisation.visual_engine import VisualEngine

# In this integrated visual pipeline we always keep PCA plotting disabled;
# PCA is still computed, but only the generative visual output is shown.
pca_runner.ENABLE_PCA_PLOT = False


MODEL_DIR = Path("AI_systems") / "latent_mapper_artifacts_live_social"
WARMUP_POINTS = 10
WARMUP_EPOCHS = 10
TRAVERSAL_STEPS = 6
RENDER_FPS = 60.0
# Buffered playback delay to absorb websocket/update jitter before rendering.
RENDER_DELAY_SEC = 0.30
MAX_RENDER_BUFFER_SEC = 3.0
MAX_ADAPTIVE_RENDER_DELAY_SEC = 1.20
ADAPTIVE_DELAY_SCALE = 0.88

# Live social integration controls.
USE_SOCIAL_LIVE = True
SOCIAL_DEBUG = False
SOCIAL_QUERY = "bitcoin OR btc"
SOCIAL_FETCH_LIMIT = 100
SOCIAL_POLL_INTERVAL_SEC = 30.0
SOCIAL_ROLLING_POSTS = 300
SOCIAL_MODEL_DIR = "AI_systems/scraper_model_artifacts"
SOCIAL_STALE_MAX_SEC = 180.0
SOCIAL_BLEND_MAX_WEIGHT = 0.55


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


class LiveBTCVisualBridge:
    def __init__(self):
        self.mapper = StreamingLatentVisualMapper(
            model_dir=MODEL_DIR,
            pca_dim=3,
            stream_buffer_size=512,
            train_window=128,
            train_every=8,
            traversal_steps=TRAVERSAL_STEPS,
        )
        self.model_ready = self.mapper.load()
        if self.model_ready:
            set_default_mapper(self.mapper)

        self.pca_warmup: Deque[np.ndarray] = deque(maxlen=max(WARMUP_POINTS * 2, 192))
        self.prev_latent: Optional[np.ndarray] = None
        self.last_visual_params: Optional[Dict[str, float]] = None
        self.last_market_condition: Dict[str, float] = {
            "turbulence": 0.5,
            "trend_bias": 0.5,
            "distortion": 0.5,
            "fragmentation": 0.5,
            "velocity": 0.5,
            "quality": 0.0,
            "social_blend_weight": 0.0,
            "social_quality": 0.0,
            "social_posts": 0.0,
            "social_age_sec": 9999.0,
            "social_stale": 1.0,
            "social_valid": 0.0,
            "social_error_streak": 0.0,
        }
        self.last_social_condition: Dict[str, float] = {
            "turbulence": 0.5,
            "trend_bias": 0.5,
            "distortion": 0.5,
            "fragmentation": 0.5,
            "velocity": 0.5,
            "quality": 0.0,
        }
        self.last_social_update_ts: float = 0.0
        self.last_social_blend_weight: float = 0.0
        self.last_social_posts: int = 0
        self.last_social_error_streak: int = 0
        self.last_regime_info: Dict[str, object] = {
            "regime_id": None,
            "confidence": 0.0,
            "n_regimes": 0,
        }
        self.render_buffer: Deque[dict] = deque(maxlen=240)
        # Track source cadence so playback delay can match real update intervals.
        self._last_source_update_ts: float = 0.0
        self._source_dt_ema: float = 1.0
        self.visual_engine = VisualEngine()

    def get_adaptive_render_delay(self) -> float:
        # Keep at least baseline delay, but follow source cadence to avoid
        # short interpolation bursts followed by long holds.
        adaptive = max(RENDER_DELAY_SEC, ADAPTIVE_DELAY_SCALE * float(self._source_dt_ema))
        return float(np.clip(adaptive, RENDER_DELAY_SEC, MAX_ADAPTIVE_RENDER_DELAY_SEC))

    @staticmethod
    def _clip01(v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))

    @staticmethod
    def _sanitize_condition(cond: Dict[str, float]) -> Dict[str, float]:
        return {
            "turbulence": _clip01(float(cond.get("turbulence", 0.5))),
            "trend_bias": _clip01(float(cond.get("trend_bias", 0.5))),
            "distortion": _clip01(float(cond.get("distortion", 0.5))),
            "fragmentation": _clip01(float(cond.get("fragmentation", 0.5))),
            "velocity": _clip01(float(cond.get("velocity", 0.5))),
            "quality": _clip01(float(cond.get("quality", 0.0))),
        }

    def update_social_state(self, snapshot: Dict[str, object]) -> None:
        factors = snapshot.get("factors", {}) if isinstance(snapshot, dict) else {}
        self.last_social_condition = self._sanitize_condition(dict(factors) if isinstance(factors, dict) else {})
        self.last_social_update_ts = float(snapshot.get("last_update_ts", self.last_social_update_ts) or self.last_social_update_ts)
        self.last_social_posts = int(snapshot.get("rolling_posts", self.last_social_posts) or self.last_social_posts)
        self.last_social_error_streak = int(snapshot.get("error_streak", self.last_social_error_streak) or self.last_social_error_streak)

    def blend_market_social_conditions(self, market_condition: Dict[str, float]) -> Dict[str, float]:
        market = self._sanitize_condition(market_condition)
        social = self._sanitize_condition(self.last_social_condition)

        social_quality = _clip01(float(social.get("quality", 0.0)))
        self.last_social_blend_weight = float(np.clip(SOCIAL_BLEND_MAX_WEIGHT * social_quality, 0.0, SOCIAL_BLEND_MAX_WEIGHT))
        w = self.last_social_blend_weight
        social_age_sec = float(max(0.0, time.time() - self.last_social_update_ts)) if self.last_social_update_ts > 0 else 9999.0
        social_stale = 1.0 if social_age_sec > SOCIAL_STALE_MAX_SEC else 0.0
        social_valid = 1.0 if (social_quality >= 0.15 and social_stale < 0.5) else 0.0

        out = {
            "turbulence": _clip01((1.0 - w) * market["turbulence"] + w * social["turbulence"]),
            "trend_bias": _clip01((1.0 - w) * market["trend_bias"] + w * social["trend_bias"]),
            "distortion": _clip01((1.0 - w) * market["distortion"] + w * social["distortion"]),
            "fragmentation": _clip01((1.0 - w) * market["fragmentation"] + w * social["fragmentation"]),
            "velocity": _clip01((1.0 - w) * market["velocity"] + w * social["velocity"]),
            "quality": _clip01((1.0 - w) * market["quality"] + w * social["quality"]),
            "social_blend_weight": float(w),
            "social_quality": float(social_quality),
            "social_posts": float(self.last_social_posts),
            "social_age_sec": float(social_age_sec),
            "social_stale": float(social_stale),
            "social_valid": float(social_valid),
            "social_error_streak": float(self.last_social_error_streak),
        }
        return out

    def build_combined_live_matrix(self, symbol: str) -> np.ndarray:
        market_matrix = rtc.build_feature_matrix(symbol)
        if market_matrix.size == 0:
            return market_matrix

        social = self._sanitize_condition(self.last_social_condition)
        social_vec = np.array(
            [
                social["turbulence"],
                social["trend_bias"],
                social["distortion"],
                social["fragmentation"],
                social["velocity"],
            ],
            dtype=np.float64,
        )
        social_block = np.repeat(social_vec.reshape(1, -1), market_matrix.shape[0], axis=0)

        # Cross terms tie social state to current market movement channels.
        # build_feature_matrix() numerical order includes:
        # [.., price_range idx=6, price_change idx=7, volume idx=4, ..]
        m_price_range = market_matrix[:, 6]
        m_price_change = market_matrix[:, 7]
        m_volume = market_matrix[:, 4]

        cross = np.column_stack(
            [
                social_block[:, 1] * m_price_change,
                social_block[:, 0] * m_price_range,
                social_block[:, 4] * m_volume,
            ]
        )

        return np.hstack([market_matrix, social_block, cross]).astype(np.float64)

    def _blend_semantic_params(self, mapper_params: Dict[str, float], market_condition: Dict[str, float]) -> Dict[str, float]:
        # Preserve mapper personality but expose theta-aligned semantic names.
        q = self._clip01(float(market_condition.get("quality", 0.0)))
        turbulence = self._clip01(float(market_condition.get("turbulence", 0.5)))
        trend_bias = self._clip01(float(market_condition.get("trend_bias", 0.5)))
        distortion = self._clip01(float(market_condition.get("distortion", 0.5)))
        fragmentation = self._clip01(float(market_condition.get("fragmentation", 0.5)))
        velocity = self._clip01(float(market_condition.get("velocity", 0.5)))

        # Increase semantic influence as data quality becomes reliable.
        w_sem = 0.25 + 0.50 * q

        # Mapper artifacts still emit legacy renderer keys; map them to semantic priors.
        p_sem = {
            "turbulence": self._clip01(float(mapper_params.get("turbulence", mapper_params.get("motion_intensity", 0.5)))),
            "trend_bias": self._clip01(float(mapper_params.get("trend_bias", mapper_params.get("color_dynamics", 0.5)))),
            "distortion": self._clip01(float(mapper_params.get("distortion", mapper_params.get("distortion_strength", 0.5)))),
            "fragmentation": self._clip01(float(mapper_params.get("fragmentation", mapper_params.get("noise_scale", 0.5)))),
            "velocity": self._clip01(float(mapper_params.get("velocity", mapper_params.get("particle_density", 0.5)))),
        }

        out = {
            "turbulence": self._clip01((1.0 - w_sem) * p_sem["turbulence"] + w_sem * turbulence),
            "trend_bias": self._clip01((1.0 - w_sem) * p_sem["trend_bias"] + w_sem * trend_bias),
            "distortion": self._clip01((1.0 - w_sem) * p_sem["distortion"] + w_sem * distortion),
            "fragmentation": self._clip01((1.0 - w_sem) * p_sem["fragmentation"] + w_sem * fragmentation),
            "velocity": self._clip01((1.0 - w_sem) * p_sem["velocity"] + w_sem * velocity),
        }
        return out

    def _enrich_live_latent(self, z_t: np.ndarray, market_condition: Dict[str, float]) -> np.ndarray:
        # Keep 3D dimensionality unchanged; inject low-amplitude semantic drift.
        z = np.asarray(z_t, dtype=np.float32).reshape(-1)
        if z.shape[0] != 3:
            return z

        turbulence = self._clip01(float(market_condition.get("turbulence", 0.5)))
        trend_bias = self._clip01(float(market_condition.get("trend_bias", 0.5)))
        distortion = self._clip01(float(market_condition.get("distortion", 0.5)))
        fragmentation = self._clip01(float(market_condition.get("fragmentation", 0.5)))
        velocity = self._clip01(float(market_condition.get("velocity", 0.5)))
        q = self._clip01(float(market_condition.get("quality", 0.0)))

        # Small semantic perturbation vector; scaled by quality to avoid startup noise.
        semantic = np.array([
            0.55 * (trend_bias - 0.5) + 0.35 * (velocity - 0.5),
            0.60 * (distortion - 0.5) + 0.25 * (fragmentation - 0.5),
            0.65 * (turbulence - 0.5) + 0.20 * (fragmentation - 0.5),
        ], dtype=np.float32)
        return (z + (0.18 + 0.22 * q) * semantic).astype(np.float32)

    @staticmethod
    def _lerp_float_dict(a: Dict[str, float], b: Dict[str, float], alpha: float) -> Dict[str, float]:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        keys = set(a.keys()) | set(b.keys())
        out: Dict[str, float] = {}
        for k in keys:
            av = float(a.get(k, b.get(k, 0.0)))
            bv = float(b.get(k, a.get(k, 0.0)))
            out[k] = float((1.0 - alpha) * av + alpha * bv)
        return out

    @staticmethod
    def _blend_regime_info(a: Dict[str, object], b: Dict[str, object], alpha: float) -> Dict[str, object]:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        regime_id = a.get("regime_id") if alpha < 0.5 else b.get("regime_id")
        conf_a = float(a.get("confidence", 0.0))
        conf_b = float(b.get("confidence", 0.0))
        n_a = int(a.get("n_regimes", 0) or 0)
        n_b = int(b.get("n_regimes", 0) or 0)
        return {
            "regime_id": regime_id,
            "confidence": float((1.0 - alpha) * conf_a + alpha * conf_b),
            "n_regimes": int(round((1.0 - alpha) * n_a + alpha * n_b)),
        }

    def _push_render_frame(self, timestamp: float, params: Dict[str, float], regime: Dict[str, object], market: Dict[str, float]) -> None:
        self.render_buffer.append(
            {
                "ts": float(timestamp),
                "params": {k: float(v) for k, v in params.items()},
                "regime": dict(regime),
                "market": {k: float(v) for k, v in market.items()},
            }
        )

        # Keep a short, recent playback buffer.
        while len(self.render_buffer) > 2:
            oldest = float(self.render_buffer[0]["ts"])
            newest = float(self.render_buffer[-1]["ts"])
            if newest - oldest <= MAX_RENDER_BUFFER_SEC:
                break
            self.render_buffer.popleft()

    def get_render_frame(self, target_ts: float) -> Optional[dict]:
        if not self.render_buffer:
            if self.last_visual_params is None:
                return None
            return {
                "params": dict(self.last_visual_params),
                "regime": dict(self.last_regime_info),
                "market": dict(self.last_market_condition),
            }

        # Advance the left bound as time moves on so we interpolate on the latest segment.
        while len(self.render_buffer) >= 2 and float(self.render_buffer[1]["ts"]) <= target_ts:
            self.render_buffer.popleft()

        if len(self.render_buffer) == 1:
            return {
                "params": dict(self.render_buffer[0]["params"]),
                "regime": dict(self.render_buffer[0]["regime"]),
                "market": dict(self.render_buffer[0]["market"]),
            }

        f0 = self.render_buffer[0]
        f1 = self.render_buffer[1]
        t0 = float(f0["ts"])
        t1 = float(f1["ts"])

        if t1 <= t0:
            return {
                "params": dict(f1["params"]),
                "regime": dict(f1["regime"]),
                "market": dict(f1["market"]),
            }

        alpha = float(np.clip((target_ts - t0) / (t1 - t0), 0.0, 1.0))
        return {
            "params": self._lerp_float_dict(f0["params"], f1["params"], alpha),
            "regime": self._blend_regime_info(f0["regime"], f1["regime"], alpha),
            "market": self._lerp_float_dict(f0["market"], f1["market"], alpha),
        }

    def _try_warmup(self):
        if self.model_ready:
            return
        if len(self.pca_warmup) < WARMUP_POINTS:
            return

        warmup_array = np.asarray(self.pca_warmup, dtype=np.float32)
        self.mapper.warmup_train(
            warmup_array,
            epochs=WARMUP_EPOCHS,
            batch_size=32,
            verbose=0,
        )
        set_default_mapper(self.mapper)
        self.model_ready = True
        print(f"[VISUAL] Mapper warmup complete with {len(warmup_array)} PCA points.")

    def on_new_latent(self, z_t: np.ndarray, market_condition: Optional[Dict[str, float]] = None) -> Optional[Dict[str, float]]:
        now_ts = time.perf_counter()
        if self._last_source_update_ts > 0.0:
            dt = float(np.clip(now_ts - self._last_source_update_ts, 1.0 / RENDER_FPS, 5.0))
            self._source_dt_ema = float(0.70 * self._source_dt_ema + 0.30 * dt)
        self._last_source_update_ts = now_ts

        z_t = np.asarray(z_t, dtype=np.float32).reshape(-1)
        if market_condition is None:
            market_condition = self.last_market_condition
        self.last_market_condition = {
            "turbulence": float(market_condition.get("turbulence", 0.5)),
            "trend_bias": float(market_condition.get("trend_bias", 0.5)),
            "distortion": float(market_condition.get("distortion", 0.5)),
            "fragmentation": float(market_condition.get("fragmentation", 0.5)),
            "velocity": float(market_condition.get("velocity", 0.5)),
            "quality": float(market_condition.get("quality", 0.0)),
            "social_blend_weight": float(market_condition.get("social_blend_weight", self.last_social_blend_weight)),
            "social_quality": float(market_condition.get("social_quality", self.last_social_condition.get("quality", 0.0))),
            "social_posts": float(market_condition.get("social_posts", self.last_social_posts)),
            "social_age_sec": float(market_condition.get("social_age_sec", 9999.0)),
            "social_stale": float(market_condition.get("social_stale", 1.0)),
            "social_valid": float(market_condition.get("social_valid", 0.0)),
            "social_error_streak": float(market_condition.get("social_error_streak", self.last_social_error_streak)),
        }

        z_in = self._enrich_live_latent(z_t, self.last_market_condition)
        self.pca_warmup.append(z_in)

        self._try_warmup()
        if not self.model_ready:
            print(f"[VISUAL] Warmup collecting PCA points: {len(self.pca_warmup)}/{WARMUP_POINTS}")
            self.prev_latent = z_in
            return None

        params = self.mapper.process_stream_step(z_in)

        if self.prev_latent is not None:
            traversal = self.mapper.traversal_parameters(self.prev_latent, z_in, steps=TRAVERSAL_STEPS)
            params = traversal[-1]

        self.prev_latent = z_in
        self.last_visual_params = self._blend_semantic_params(params, self.last_market_condition)
        self.last_regime_info = self.mapper.get_latest_regime_info()
        self._push_render_frame(
            timestamp=now_ts,
            params=self.last_visual_params,
            regime=self.last_regime_info,
            market=self.last_market_condition,
        )
        return self.last_visual_params


async def stream_btc_visual_parameters() -> None:
    print("=" * 58)
    print("  Live BTC -> PCA -> Generative Visual Parameters")
    print(f"  Symbol   : {rtc.SYMBOLS[0]}")
    print(f"  Interval : {rtc.INTERVAL}")
    print(f"  PCA Plot : {'ON' if pca_runner.ENABLE_PCA_PLOT else 'OFF'}")
    print(f"  Social   : {'ON' if USE_SOCIAL_LIVE else 'OFF'}")
    if USE_SOCIAL_LIVE:
        print(f"  SocialQ  : {SOCIAL_QUERY}")
        print(f"  SocialN  : {SOCIAL_FETCH_LIMIT} per {int(SOCIAL_POLL_INTERVAL_SEC)}s")
        print(f"  SDebug   : {'ON' if SOCIAL_DEBUG else 'OFF'}")
    print("=" * 58)

    rtc.preload_buffers_from_csv(rtc.OUTPUT_DIR, rtc.BUFFER_SIZE)
    bridge = LiveBTCVisualBridge()

    stop_render = asyncio.Event()

    social_poller: Optional[LiveSocialSentimentPoller] = None
    social_task: Optional[asyncio.Task] = None
    if USE_SOCIAL_LIVE:
        social_poller = LiveSocialSentimentPoller(
            query=SOCIAL_QUERY,
            fetch_limit=SOCIAL_FETCH_LIMIT,
            rolling_posts=SOCIAL_ROLLING_POSTS,
            use_ai_model=True,
            model_dir=SOCIAL_MODEL_DIR,
            debug=SOCIAL_DEBUG,
            handle=os.getenv("BLUESKY_HANDLE"),
            password=os.getenv("BLUESKY_APP_PASSWORD"),
        )

    async def social_loop() -> None:
        if social_poller is None:
            return
        while not stop_render.is_set():
            snapshot = await asyncio.to_thread(social_poller.get_snapshot)
            if snapshot:
                bridge.update_social_state(snapshot)

            snapshot = await asyncio.to_thread(social_poller.poll_once)
            if snapshot:
                bridge.update_social_state(social_poller.get_snapshot())

            if SOCIAL_DEBUG:
                age = max(0.0, time.time() - float(bridge.last_social_update_ts)) if bridge.last_social_update_ts > 0 else -1.0
                print(
                    "[SOCIAL-LOOP] "
                    f"posts={bridge.last_social_posts} "
                    f"err={bridge.last_social_error_streak} "
                    f"age={age:.1f}s "
                    f"social_quality={bridge.last_social_condition['quality']:.3f}"
                )

            await asyncio.sleep(SOCIAL_POLL_INTERVAL_SEC)

    async def render_loop() -> None:
        frame_dt = 1.0 / max(1.0, RENDER_FPS)
        while not stop_render.is_set():
            render_delay = bridge.get_adaptive_render_delay()
            render_state = bridge.get_render_frame(time.perf_counter() - render_delay)
            if render_state is not None:
                bridge.visual_engine.render(
                    render_state["params"],
                    regime_info=render_state["regime"],
                    market_condition=render_state["market"],
                )
            bridge.visual_engine.pump_events()
            await asyncio.sleep(frame_dt)

    render_task = asyncio.create_task(render_loop())
    if USE_SOCIAL_LIVE and social_poller is not None:
        social_task = asyncio.create_task(social_loop())

    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)

    try:
        async with bm.kline_socket(symbol=rtc.SYMBOLS[0], interval=rtc.INTERVAL) as stream:
            while True:
                try:
                    msg = await asyncio.wait_for(stream.recv(), timeout=0.08)
                except asyncio.TimeoutError:
                    continue

                if msg.get("e") == "error":
                    print(f"[ERROR] {rtc.SYMBOLS[0]}: {msg.get('m', 'unknown error')}")
                    break

                kline = msg["k"]
                row = rtc.extract_features(kline, rtc.SYMBOLS[0])
                if row is None:
                    continue

                rtc.buffers[rtc.SYMBOLS[0]].append(row)
                rtc.save_row_to_csv(row, rtc.SYMBOLS[0], rtc.OUTPUT_DIR)

                matrix = bridge.build_combined_live_matrix(rtc.SYMBOLS[0])
                pca_latest = pca_runner.run_pca(rtc.SYMBOLS[0], matrix)
                market_condition = rtc.build_market_condition_factors(rtc.SYMBOLS[0])
                blended_condition = bridge.blend_market_social_conditions(market_condition)

                if pca_latest is None:
                    min_rows = max(pca_runner.PCA_N_COMPONENTS + 1, 2)
                    print(f"[{rtc.SYMBOLS[0]}] PCA waiting: need >= {min_rows}, have {matrix.shape[0]}")
                    continue

                visual_params = bridge.on_new_latent(np.asarray(pca_latest, dtype=np.float32), market_condition=blended_condition)
                if visual_params is None:
                    continue
                regime_id = bridge.last_regime_info.get("regime_id")
                regime_conf = float(bridge.last_regime_info.get("confidence", 0.0))
                n_regimes = int(bridge.last_regime_info.get("n_regimes", 0))
                regime_label = (
                    f"R{regime_id}/{n_regimes} conf={regime_conf:.3f}"
                    if regime_id is not None and n_regimes > 0
                    else "R?/?: warming"
                )
                print(
                    f"[VISUAL] {', '.join(f'{k}={v:.3f}' for k, v in visual_params.items())} | "
                    f"T={bridge.last_market_condition['turbulence']:.3f} "
                    f"B={bridge.last_market_condition['trend_bias']:.3f} "
                    f"D={bridge.last_market_condition['distortion']:.3f} "
                    f"F={bridge.last_market_condition['fragmentation']:.3f} "
                    f"V={bridge.last_market_condition['velocity']:.3f} "
                    f"Q={bridge.last_market_condition['quality']:.3f} "
                    f"SW={bridge.last_social_blend_weight:.3f} "
                    f"SP={bridge.last_social_posts} "
                    f"SQUAL={bridge.last_social_condition['quality']:.3f} | {regime_label}"
                )

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
    finally:
        stop_render.set()
        render_task.cancel()
        if social_task is not None:
            social_task.cancel()
        try:
            await render_task
        except asyncio.CancelledError:
            pass
        if social_task is not None:
            try:
                await social_task
            except asyncio.CancelledError:
                pass
        bridge.mapper.save()
        await client.close_connection()
        print("[DONE] Connection closed and mapper state saved.")


async def main() -> None:
    while True:
        try:
            await stream_btc_visual_parameters()
            break
        except Exception as exc:
            print(f"[WARN] Visual stream error: {exc}. Reconnecting in 5s...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
