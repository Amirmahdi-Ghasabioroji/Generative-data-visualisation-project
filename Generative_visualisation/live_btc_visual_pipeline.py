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
from pathlib import Path
from typing import Deque, Dict, Optional
import sys

import numpy as np
from binance import AsyncClient, BinanceSocketManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Data_Pipeline import Real_time_Crypto as rtc
from Generative_visualisation import live_btc_pca_visual as pca_runner
from AI_systems.latent_visual_mapper import (
    StreamingLatentVisualMapper,
    set_default_mapper,
)
from Generative_visualisation.visual_engine import VisualEngine

# In this integrated visual pipeline we always keep PCA plotting disabled;
# PCA is still computed, but only the generative visual output is shown.
pca_runner.ENABLE_PCA_PLOT = False


MODEL_DIR = Path("AI_systems") / "latent_mapper_artifacts"
WARMUP_POINTS = 96
WARMUP_EPOCHS = 30
TRAVERSAL_STEPS = 6


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
        self.last_regime_info: Dict[str, object] = {
            "regime_id": None,
            "confidence": 0.0,
            "n_regimes": 0,
        }
        self.visual_engine = VisualEngine()

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

    def on_new_latent(self, z_t: np.ndarray) -> Optional[Dict[str, float]]:
        z_t = np.asarray(z_t, dtype=np.float32).reshape(-1)
        self.pca_warmup.append(z_t)

        self._try_warmup()
        if not self.model_ready:
            print(f"[VISUAL] Warmup collecting PCA points: {len(self.pca_warmup)}/{WARMUP_POINTS}")
            self.prev_latent = z_t
            return None

        params = self.mapper.process_stream_step(z_t)

        if self.prev_latent is not None:
            traversal = self.mapper.traversal_parameters(self.prev_latent, z_t, steps=TRAVERSAL_STEPS)
            params = traversal[-1]

        self.prev_latent = z_t
        self.last_regime_info = self.mapper.get_latest_regime_info()
        return params


async def stream_btc_visual_parameters() -> None:
    print("=" * 58)
    print("  Live BTC -> PCA -> Generative Visual Parameters")
    print(f"  Symbol   : {rtc.SYMBOLS[0]}")
    print(f"  Interval : {rtc.INTERVAL}")
    print(f"  PCA Plot : {'ON' if pca_runner.ENABLE_PCA_PLOT else 'OFF'}")
    print("=" * 58)

    rtc.preload_buffers_from_csv(rtc.OUTPUT_DIR, rtc.BUFFER_SIZE)
    bridge = LiveBTCVisualBridge()

    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)

    try:
        async with bm.kline_socket(symbol=rtc.SYMBOLS[0], interval=rtc.INTERVAL) as stream:
            while True:
                msg = await stream.recv()

                if msg.get("e") == "error":
                    print(f"[ERROR] {rtc.SYMBOLS[0]}: {msg.get('m', 'unknown error')}")
                    break

                kline = msg["k"]
                row = rtc.extract_features(kline, rtc.SYMBOLS[0])
                if row is None:
                    continue

                rtc.buffers[rtc.SYMBOLS[0]].append(row)
                rtc.save_row_to_csv(row, rtc.SYMBOLS[0], rtc.OUTPUT_DIR)

                matrix = rtc.build_feature_matrix(rtc.SYMBOLS[0])
                pca_latest = pca_runner.run_pca(rtc.SYMBOLS[0], matrix)

                if pca_latest is None:
                    min_rows = max(pca_runner.PCA_N_COMPONENTS + 1, 2)
                    print(f"[{rtc.SYMBOLS[0]}] PCA waiting: need >= {min_rows}, have {matrix.shape[0]}")
                    continue

                visual_params = bridge.on_new_latent(np.asarray(pca_latest, dtype=np.float32))
                if visual_params is None:
                    continue

                bridge.visual_engine.render(visual_params)
                regime_id = bridge.last_regime_info.get("regime_id")
                regime_conf = float(bridge.last_regime_info.get("confidence", 0.0))
                n_regimes = int(bridge.last_regime_info.get("n_regimes", 0))
                regime_label = (
                    f"R{regime_id}/{n_regimes} conf={regime_conf:.3f}"
                    if regime_id is not None and n_regimes > 0
                    else "R?/?: warming"
                )
                print(
                    f"[VISUAL] {', '.join(f'{k}={v:.3f}' for k, v in visual_params.items())} | {regime_label}"
                )

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
    finally:
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
