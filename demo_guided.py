import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append("notebook")
from inference import Inference, load_image  # noqa: E402

from guidance import (  # noqa: E402  -- diki's modules at repo root
    CompositeGuidance,
    DepthGuidance,
    NormalGuidance,
    PoseGuidance,
    ShapeGuidance,
)
from guided_solver import GuidedEuler, GuidedEulerConfig  # noqa: E402


def load_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 0).astype(np.float32)


def fetch_intrinsics(pipeline, image, mask):
    """
    Run MoGe via pipeline.compute_pointmap to get K. Same trick as jason's
    original demo_guided.py — pre-compute the pointmap once externally so
    we have intrinsics before constructing GuidedEuler.

    Returns K as a 3x3 tensor in NORMALIZED image coords ([0,1] range),
    which is the format diki's modules expect (they multiply by image_size
    when building the PyTorch3D camera).
    """
    # Wrap as RGBA the way the pipeline does (mask in alpha channel).
    image_merged = pipeline.merge_image_and_mask(image, mask)
    pointmap_dict = pipeline.compute_pointmap(image_merged, None)
    K = pointmap_dict.get("intrinsics", None)
    if K is None:
        raise RuntimeError(
            "compute_pointmap returned no 'intrinsics'. "
            "Check MoGe depth model is loaded correctly."
        )
    if torch.is_tensor(K):
        K = K.float()
    else:
        K = torch.tensor(K, dtype=torch.float32)
    if K.ndim == 3:
        K = K[0]
    if K.shape != (3, 3):
        raise ValueError(f"K should be 3x3, got {tuple(K.shape)}")
    # diki's modules expect K in normalized coords ([0,1]). MoGe returns it
    # that way already; if your depth model returns pixel-coord K, divide
    # rows by image H/W here.
    return K


def build_composite(args):
    modules = []
    if args.ss_scale > 0:
        print(f"[GUIDANCE] ShapeGuidance   scale={args.ss_scale}  start_t={args.shape_start_t}")
        modules.append(ShapeGuidance(
            mask_path=args.mask,
            shape_scale=args.ss_scale,
            start_t=args.shape_start_t,
        ))
    if args.pose_scale > 0:
        print(f"[GUIDANCE] PoseGuidance    scale={args.pose_scale}  start_t={args.pose_start_t}")
        modules.append(PoseGuidance(
            mask_path=args.mask,
            pose_scale=args.pose_scale,
            start_t=args.pose_start_t,
            w_centroid=args.w_centroid,
            w_size=args.w_size,
        ))
    if args.depth_scale > 0:
        if args.depth is None:
            print("[GUIDANCE] WARNING: --depth-scale set but --depth missing; skipping DepthGuidance")
        else:
            print(f"[GUIDANCE] DepthGuidance   scale={args.depth_scale}  start_t={args.depth_start_t}")
            modules.append(DepthGuidance(
                depth_path=args.depth,
                depth_scale=args.depth_scale,
                start_t=args.depth_start_t,
            ))
    if args.normal_scale > 0:
        if args.depth is None:
            print("[GUIDANCE] WARNING: --normal-scale set but --depth missing; skipping NormalGuidance")
        else:
            print(f"[GUIDANCE] NormalGuidance  scale={args.normal_scale}  start_t={args.normal_start_t}")
            modules.append(NormalGuidance(
                depth_path=args.depth,
                normal_scale=args.normal_scale,
                start_t=args.normal_start_t,
            ))
    if not modules:
        return None
    return CompositeGuidance(modules)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mask",  required=True)
    p.add_argument("--depth", default=None, help="GT depth .npy (Open3DHOI format)")
    p.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    p.add_argument("--out", default="outputs/merged/splat.ply")
    p.add_argument("--seed", type=int, default=42)
    # Per-module scales (diki convention)
    p.add_argument("--ss-scale",     type=float, default=0.0)
    p.add_argument("--pose-scale",   type=float, default=0.0)
    p.add_argument("--depth-scale",  type=float, default=0.0)
    p.add_argument("--normal-scale", type=float, default=0.0)
    p.add_argument("--w-centroid", type=float, default=1.0)
    p.add_argument("--w-size",     type=float, default=1.0)
    # Per-module start_t (when each module becomes active)
    p.add_argument("--shape-start-t",  type=float, default=0.8)
    p.add_argument("--pose-start-t",   type=float, default=0.9)
    p.add_argument("--depth-start-t",  type=float, default=0.7)
    p.add_argument("--normal-start-t", type=float, default=0.7)
    # Solver-level scheduling (jason convention)
    p.add_argument("--start-t", type=float, default=0.95)
    p.add_argument("--end-t",   type=float, default=0.05)
    p.add_argument("--ramp",    default="constant", choices=["constant", "triangle", "cosine"])
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"[demo] loading model: {args.config}")
    inference = Inference(args.config, compile=False)
    pipeline = inference._pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[demo] loading image / mask")
    image = load_image(args.image)
    mask  = load_mask(args.mask)

    composite = build_composite(args)

    if composite is not None:
        print(f"[demo] computing intrinsics via MoGe...")
        K = fetch_intrinsics(pipeline, image, mask).to(device)
        print(f"[demo] K=\n{K.cpu().numpy()}")

        pipeline.guided_solver = GuidedEuler(
            guidance=composite,
            ss_decoder=pipeline.models["ss_decoder"],
            pose_decoder=pipeline.pose_decoder,
            intrinsics=K,
            config=GuidedEulerConfig(
                start_t=args.start_t,
                end_t=args.end_t,
                ramp=args.ramp,
                verbose=args.verbose,
            ),
        )
        # scene_scale / scene_shift injected by pipeline patch just-in-time.
        print(f"[demo] guidance ON   window=[{args.end_t}, {args.start_t}]  ramp={args.ramp}")
    else:
        if hasattr(pipeline, "guided_solver"):
            delattr(pipeline, "guided_solver")
        print("[demo] guidance OFF  (all scales are 0)")

    print("[demo] sampling...")
    output = inference(image, mask, seed=args.seed)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    if "gs" in output and hasattr(output["gs"], "save_ply"):
        output["gs"].save_ply(args.out)
        print(f"[demo] saved -> {args.out}")
    else:
        print(f"[demo] no 'gs' in output; keys: {list(output.keys())}")


if __name__ == "__main__":
    main()