from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sam3d_objects.model.backbone.generator.flow_matching.solver import Euler


@dataclass
class GuidedEulerConfig:
    """Solver-level scheduling. Per-module scales live on the modules themselves."""
    start_t: float = 0.95          # guidance active when t < start_t
    end_t:   float = 0.05          # guidance active when t > end_t
    ramp:    str   = "constant"    # "constant" | "triangle" | "cosine"
    verbose: bool  = False


class GuidedEuler(Euler):
    """
    Drop-in Euler subclass. After each standard step, optionally calls
    a ``guidance`` object (typically diki's ``CompositeGuidance``) and
    writes its corrections back into ``x_t`` in place.

    External setup pattern (matches jason's demo_guided.py)::

        from sam3d_objects.pipeline.guidance import (
            CompositeGuidance, ShapeGuidance, PoseGuidance,
            DepthGuidance, NormalGuidance,
        )
        from guided_solver import GuidedEuler, GuidedEulerConfig

        composite = CompositeGuidance([
            ShapeGuidance(mask_path, shape_scale=5.0, start_t=0.8),
            PoseGuidance(mask_path,  pose_scale=0.05, start_t=0.9),
            DepthGuidance(depth_path, depth_scale=5.0, start_t=0.7),
            NormalGuidance(depth_path, normal_scale=2.0, start_t=0.7),
        ])

        pipeline.guided_solver = GuidedEuler(
            guidance=composite,
            ss_decoder=pipeline.models["ss_decoder"],
            pose_decoder=pipeline.pose_decoder,
            intrinsics=K,                      # 3x3 normalized
            config=GuidedEulerConfig(start_t=0.95, end_t=0.05, ramp="constant"),
        )
        # scene_scale / scene_shift are injected by the pipeline patch
        # just-in-time before sampling (they only exist after preprocessor).

        output = inference(image, mask, seed=42)
    """

    def __init__(
        self,
        guidance,
        *,
        ss_decoder=None,
        pose_decoder=None,
        intrinsics: Optional[torch.Tensor] = None,
        config: GuidedEulerConfig = GuidedEulerConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.guidance = guidance
        self.cfg = config

        # Context for guidance.apply(). Partially populated at construction;
        # scene_scale / scene_shift come from the pipeline patch (they exist
        # only after preprocessor runs inside sample_sparse_structure).
        self.context: dict[str, Any] = {
            "ss_decoder":   ss_decoder,
            "pose_decoder": pose_decoder,
            "intrinsics":   intrinsics,
            "scene_scale":  None,
            "scene_shift":  None,
        }

    # ------------------------------------------------------------------
    # Solver overrides
    # ------------------------------------------------------------------

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        # 1) Standard Euler step.
        x_tp1 = super().step(dynamics_fn, x_t, t, dt, *args, **kwargs)

        # 2) Time-window gate.
        if self.guidance is None:
            return x_tp1
        t_val = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
        ramp = self._ramp_at(t_val)
        if ramp <= 0.0:
            return x_tp1

        # 3) Verify we have what diki's modules need.
        ctx = self.context
        if ctx.get("ss_decoder") is None or ctx.get("pose_decoder") is None:
            if self.cfg.verbose:
                print(f"[GuidedEuler t={t_val:.3f}] missing ss_decoder or "
                      f"pose_decoder in context — skipping")
            return x_tp1
        if ctx.get("intrinsics") is None:
            if self.cfg.verbose:
                print(f"[GuidedEuler t={t_val:.3f}] missing intrinsics — skipping")
            return x_tp1

        # 4) Non-dict latent: wrap (diki's contract is dict-based).
        wrapped = x_tp1 if isinstance(x_tp1, dict) else {"shape": x_tp1}

        # 5) Dispatch.
        corrections = self.guidance.apply(
            wrapped,
            ctx["ss_decoder"],
            ctx["pose_decoder"],
            ctx["intrinsics"],
            scene_scale=ctx.get("scene_scale"),
            scene_shift=ctx.get("scene_shift"),
            t_step=t_val,
        )
        if not corrections:
            return x_tp1

        # 6) Write-back with ramp interpolation:
        #       x ← original + ramp * (corrected − original)
        #    ramp=1 → full correction, ramp=0 → no change.
        for key, corrected in corrections.items():
            if key not in wrapped:
                if self.cfg.verbose:
                    print(f"[GuidedEuler t={t_val:.3f}] guidance returned unknown "
                          f"key '{key}' (have: {list(wrapped.keys())})")
                continue
            tgt = wrapped[key]
            corrected = corrected.to(tgt.device).to(tgt.dtype)
            tgt.data.copy_(tgt.data + ramp * (corrected - tgt.data))

        if self.cfg.verbose:
            print(f"[GuidedEuler t={t_val:.3f}] applied {len(corrections)} "
                  f"correction(s) with ramp={ramp:.3f}")

        return x_tp1

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _ramp_at(self, t_val: float) -> float:
        """Returns ramp factor in [0,1]. 0 outside the [end_t, start_t] window."""
        if not (self.cfg.end_t < t_val < self.cfg.start_t):
            return 0.0
        span = max(self.cfg.start_t - self.cfg.end_t, 1e-8)
        u = (t_val - self.cfg.end_t) / span  # 0 at end_t, 1 at start_t

        if self.cfg.ramp == "constant":
            return 1.0
        if self.cfg.ramp == "triangle":
            return 1.0 - 2.0 * abs(u - 0.5)
        if self.cfg.ramp == "cosine":
            return 0.5 * (1.0 - math.cos(2.0 * math.pi * u))
        raise ValueError(f"unknown ramp: {self.cfg.ramp!r}")