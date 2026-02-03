#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 (GitHub) Video Tracker + PrototypeBank(global IDs across sequences)
Cambridge RGB naming: seq2_frame00001.png

B-route pipeline:
1) group frames by seqX
2) first frame: SAM2 automatic mask generation -> init masks
3) init tracker: convert init masks -> boxes -> add_new_points_or_box(frame_idx=0)
4) propagate_in_video(state) -> per-frame (object_ids, masks)
5) build track-level embedding (default: geometric embedding; easy to replace later)
6) PrototypeBank: track_id -> global_id (cross-seq consistent IDs)
7) save per-frame label PNG (uint8/uint16), optional tiny viz

Requires (besides common deps):
  - your local SAM2 repo installed/importable as `sam2`
  - checkpoints + config from SAM2 repo:
      checkpoint = ./checkpoints/sam2.1_hiera_large.pt
      model_cfg  = configs/sam2.1/sam2.1_hiera_l.yaml

Install common deps:
  pip install torch torchvision opencv-python pillow tqdm numpy

Run example:
  python sam2_cambridge_global_id.py \
    --dataset Cambridge_OldHospital --split train \
    --sam2_ckpt ./checkpoints/sam2.1_hiera_large.pt \
    --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
    --device cuda --use_fp16 true --viz true

Notes:
- This script uses SAM2AutomaticMaskGenerator for first-frame proposals.
  If your repo's import path differs, the script will try a few common fallbacks.
- Track embedding is geometric by default (fast, runs everywhere). For stronger cross-seq ID,
  replace `mask_geom_embedding()` with a SAM2 feature pooling embedding later.
"""

import os
import re
import glob
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import cv2

import tempfile
import shutil


# =========================
# Config
# =========================
@dataclass
class Config:
    image_dir: str = "datasets/Cambridge_OldHospital/train/rgb"
    out_dir: str = "datasets/Cambridge_OldHospital/train/semantics"
    dataset: str = ""
    split: str = "train"

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True  # uses bf16 autocast on CUDA

    # SAM2
    sam2_ckpt: str = "./checkpoints/sam2.1_hiera_large.pt"
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # First-frame init masks (auto mask generator)
    init_min_area: int = 300
    init_max_masks: int = 80
    init_nms_iou: float = 0.85

    # Tracking output filter
    track_min_area: int = 200
    track_score_thresh: float = 0.0  # propagate often has no score; kept for compatibility

    # PrototypeBank
    proto_sim_thresh: float = 0.75
    proto_ema: float = 0.98
    proto_max: int = 8192

    reuse_bank: bool = False
    bank_path: str = ""  # default: out_dir/proto_bank.npy

    # Output
    png_compress: int = 3
    save_uint16_if_needed: bool = True

    # Visualization (tiny)
    viz: bool = False
    viz_size: int = 96
    viz_overlay: bool = False
    viz_alpha: float = 0.55


def _str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args_to_cfg() -> Config:
    p = argparse.ArgumentParser(
        description="SAM2 tracker + PrototypeBank(global IDs) -> label PNGs per frame (seqX_frameXXXXX.png)"
    )
    p.add_argument("--dataset", type=str, default=Config.dataset)
    p.add_argument("--split", type=str, default=Config.split, choices=["train", "test"])

    p.add_argument("--image_dir", type=str, default=Config.image_dir)
    p.add_argument("--out_dir", type=str, default=Config.out_dir)

    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--device", type=str, default=Config.device)
    p.add_argument("--use_fp16", type=_str2bool, default=Config.use_fp16)

    p.add_argument("--sam2_ckpt", type=str, default=Config.sam2_ckpt)
    p.add_argument("--sam2_config", type=str, default=Config.sam2_config)

    p.add_argument("--init_min_area", type=int, default=Config.init_min_area)
    p.add_argument("--init_max_masks", type=int, default=Config.init_max_masks)
    p.add_argument("--init_nms_iou", type=float, default=Config.init_nms_iou)

    p.add_argument("--track_min_area", type=int, default=Config.track_min_area)
    p.add_argument("--track_score_thresh", type=float, default=Config.track_score_thresh)

    p.add_argument("--proto_sim_thresh", type=float, default=Config.proto_sim_thresh)
    p.add_argument("--proto_ema", type=float, default=Config.proto_ema)
    p.add_argument("--proto_max", type=int, default=Config.proto_max)

    p.add_argument("--reuse_bank", type=_str2bool, default=Config.reuse_bank)
    p.add_argument("--bank_path", type=str, default=Config.bank_path)

    p.add_argument("--png_compress", type=int, default=Config.png_compress)
    p.add_argument("--save_uint16_if_needed", type=_str2bool, default=Config.save_uint16_if_needed)

    p.add_argument("--viz", type=_str2bool, default=Config.viz)
    p.add_argument("--viz_size", type=int, default=Config.viz_size)
    p.add_argument("--viz_overlay", type=_str2bool, default=Config.viz_overlay)
    p.add_argument("--viz_alpha", type=float, default=Config.viz_alpha)

    args = p.parse_args()

    if args.dataset:
        default_image_dir = f"datasets/{args.dataset}/{args.split}/rgb"
        default_out_dir = f"datasets/{args.dataset}/{args.split}/semantics"
        if args.image_dir == Config.image_dir:
            args.image_dir = default_image_dir
        if args.out_dir == Config.out_dir:
            args.out_dir = default_out_dir

    return Config(**vars(args))


# =========================
# PrototypeBank
# =========================
class PrototypeBank:
    def __init__(self, dim: int, sim_thresh: float, ema: float, max_k: int, device: str):
        self.dim = int(dim)
        self.sim_thresh = float(sim_thresh)
        self.ema = float(ema)
        self.max_k = int(max_k)
        self.device = device
        self.P = torch.empty((0, self.dim), dtype=torch.float32, device=self.device)
        self.counts = torch.empty((0,), dtype=torch.int64, device=self.device)

    @property
    def K(self) -> int:
        return int(self.P.shape[0])

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def match_or_create(self, g: torch.Tensor) -> Tuple[int, float]:
        g = self._normalize(g.view(1, -1)).squeeze(0)

        if self.K == 0:
            self.P = g.view(1, -1).clone()
            self.counts = torch.ones((1,), dtype=torch.int64, device=self.device)
            return 0, 1.0

        Pn = self._normalize(self.P)
        sim = (Pn * g.unsqueeze(0)).sum(dim=1)
        best_sim, best_id = sim.max(dim=0)
        best_sim_f = float(best_sim.item())
        best_id_i = int(best_id.item())

        if best_sim_f >= self.sim_thresh:
            self.P[best_id_i] = self.ema * self.P[best_id_i] + (1.0 - self.ema) * g
            self.counts[best_id_i] += 1
            return best_id_i, best_sim_f

        if self.K < self.max_k:
            self.P = torch.cat([self.P, g.view(1, -1)], dim=0)
            self.counts = torch.cat(
                [self.counts, torch.ones((1,), dtype=torch.int64, device=self.device)], dim=0
            )
            return self.K - 1, 1.0

        # bank full: force assign to nearest
        self.P[best_id_i] = self.ema * self.P[best_id_i] + (1.0 - self.ema) * g
        self.counts[best_id_i] += 1
        return best_id_i, best_sim_f

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        obj = {
            "P": self.P.detach().cpu().numpy().astype(np.float32),
            "counts": self.counts.detach().cpu().numpy().astype(np.int64),
            "dim": self.dim,
            "sim_thresh": self.sim_thresh,
            "ema": self.ema,
            "max_k": self.max_k,
        }
        np.save(path, obj, allow_pickle=True)

    @staticmethod
    def load(path: str, device: str) -> "PrototypeBank":
        obj = np.load(path, allow_pickle=True).item()
        bank = PrototypeBank(
            dim=int(obj["dim"]),
            sim_thresh=float(obj.get("sim_thresh", 0.75)),
            ema=float(obj.get("ema", 0.95)),
            max_k=int(obj.get("max_k", 4096)),
            device=device,
        )
        bank.P = torch.from_numpy(obj["P"]).to(device=device, dtype=torch.float32)
        bank.counts = torch.from_numpy(obj["counts"]).to(device=device, dtype=torch.int64)
        return bank


# =========================
# Cambridge grouping: seq2_frame00001.png
# =========================
_SEQ_RE = re.compile(r"^(seq\d+)_frame(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE)


def group_frames_by_seq(image_dir: str) -> Dict[str, List[str]]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(image_dir, e)))
    paths = sorted(paths)

    seq_map: Dict[str, List[Tuple[int, str]]] = {}
    for p in paths:
        name = os.path.basename(p)
        m = _SEQ_RE.match(name)
        if not m:
            continue
        seq = m.group(1)
        frame_idx = int(m.group(2))
        seq_map.setdefault(seq, []).append((frame_idx, p))

    out: Dict[str, List[str]] = {}
    for seq, items in seq_map.items():
        items.sort(key=lambda x: x[0])
        out[seq] = [p for _, p in items]
    return out

def prepare_jpeg_sequence_from_png(frame_paths: List[str], quality: int = 95) -> Tuple[str, List[str]]:
    """
    Convert ordered PNG frames into a temp JPEG folder that SAM2 expects.
    SAM2 requires JPEG filenames to be numeric (e.g., 000000.jpg).
    Returns:
        jpeg_dir: temp folder path
        orig_paths: same as input, for mapping frame_idx -> original png path
    """
    jpeg_dir = tempfile.mkdtemp(prefix="sam2_seq_")

    for i, p in enumerate(frame_paths):
        img = Image.open(p).convert("RGB")
        out_name = f"{i:06d}.jpg"   # MUST be numeric-only basename
        out_p = os.path.join(jpeg_dir, out_name)
        img.save(out_p, "JPEG", quality=quality, subsampling=0)

    return jpeg_dir, frame_paths


# =========================
# Mask utils
# =========================
def mask_area(mask: np.ndarray) -> int:
    return int(mask.astype(np.uint8).sum())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-6)


def nms_masks(masks: List[np.ndarray], scores: List[float], iou_th: float) -> List[int]:
    order = np.argsort(np.array(scores))[::-1].tolist()
    keep = []
    for i in order:
        ok = True
        for j in keep:
            if mask_iou(masks[i], masks[j]) >= iou_th:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def _mask_to_xyxy(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return np.array([x0, y0, x1, y1], dtype=np.float32)


# =========================
# Write label map / save
# =========================
def write_masks_to_label_map(
    H: int,
    W: int,
    masks: List[np.ndarray],
    ids: List[int],
    scores: List[float],
    min_area: int = 0,
) -> np.ndarray:
    full_label = np.full((H, W), -1, dtype=np.int32)
    full_score = np.full((H, W), -1e9, dtype=np.float32)

    order = np.argsort(np.array(scores))[::-1].tolist()
    for k in order:
        m = masks[k].astype(bool)
        if min_area > 0 and int(m.sum()) < min_area:
            continue
        s = float(scores[k])
        lab = int(ids[k])
        upd = np.logical_and(m, s > full_score)
        full_label[upd] = lab
        full_score[upd] = s
    return full_label


def save_full_label_png(cfg: Config, save_path: str, label_hw: np.ndarray, max_label_id: int):
    if cfg.save_uint16_if_needed and max_label_id > 255:
        out = label_hw.astype(np.uint16)
    else:
        out = label_hw.astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, out, [cv2.IMWRITE_PNG_COMPRESSION, cfg.png_compress])


def _label_to_color(label: np.ndarray) -> np.ndarray:
    x = label.astype(np.uint32)
    r = (x * 123457 + 890123) & 255
    g = (x * 765431 + 12345) & 255
    b = (x * 345679 + 54321) & 255
    return np.stack([b, g, r], axis=-1).astype(np.uint8)  # BGR


def save_tiny_viz(cfg: Config, img_rgb: Image.Image, label_hw: np.ndarray, save_path: str):
    lab_t = torch.from_numpy(label_hw[None, None].astype(np.float32))
    lab_small = (
        F.interpolate(lab_t, size=(cfg.viz_size, cfg.viz_size), mode="nearest")
        .squeeze()
        .numpy()
        .astype(np.int32)
    )
    color = _label_to_color(lab_small)

    if cfg.viz_overlay:
        rgb = np.array(img_rgb.resize((cfg.viz_size, cfg.viz_size), resample=Image.BILINEAR))[:, :, ::-1]
        out = (cfg.viz_alpha * color.astype(np.float32) + (1 - cfg.viz_alpha) * rgb.astype(np.float32))
        out = out.clip(0, 255).astype(np.uint8)
    else:
        out = color

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3])


# =========================
# Track embedding (default: geometric)
# =========================
def mask_geom_embedding(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    area = m.sum() / (H * W + 1e-6)

    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return np.zeros((10,), dtype=np.float32)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = xs.mean() / (W + 1e-6)
    cy = ys.mean() / (H + 1e-6)
    bw = (x1 - x0 + 1) / (W + 1e-6)
    bh = (y1 - y0 + 1) / (H + 1e-6)
    aspect = bw / (bh + 1e-6)
    fill = float(m.sum()) / float((x1 - x0 + 1) * (y1 - y0 + 1) + 1e-6)

    dx = (xs - xs.mean()) / (W + 1e-6)
    dy = (ys - ys.mean()) / (H + 1e-6)
    varx = float((dx * dx).mean())
    vary = float((dy * dy).mean())

    v = np.array([area, cx, cy, bw, bh, aspect, fill, varx, vary, 1.0], dtype=np.float32)
    return v


# =========================
# SAM2 build + automask
# =========================
def build_sam2_predictors(cfg: Config):
    # GitHub example API you provided:
    # from sam2.build_sam import build_sam2, build_sam2_video_predictor
    # predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    # image_model = build_sam2(model_cfg, checkpoint)
    from sam2.build_sam import build_sam2, build_sam2_video_predictor

    video_predictor = build_sam2_video_predictor(cfg.sam2_config, cfg.sam2_ckpt)
    video_predictor = video_predictor.to(cfg.device).eval()

    image_model = build_sam2(cfg.sam2_config, cfg.sam2_ckpt)
    image_model = image_model.to(cfg.device).eval()

    # Try common import paths for SAM2AutomaticMaskGenerator
    SAM2AutomaticMaskGenerator = None
    import_errors = []

    for mod_path in [
        "sam2.automatic_mask_generator",
        "sam2.sam2_automatic_mask_generator",
        "sam2.utils.automatic_mask_generator",
    ]:
        try:
            mod = __import__(mod_path, fromlist=["SAM2AutomaticMaskGenerator"])
            SAM2AutomaticMaskGenerator = getattr(mod, "SAM2AutomaticMaskGenerator")
            break
        except Exception as e:
            import_errors.append((mod_path, str(e)))

    if SAM2AutomaticMaskGenerator is None:
        msg = "Cannot import SAM2AutomaticMaskGenerator. Tried:\n"
        msg += "\n".join([f"  - {m}: {err}" for m, err in import_errors])
        msg += "\nPlease search in your SAM2 repo for 'class SAM2AutomaticMaskGenerator' and adjust the import list."
        raise ImportError(msg)

    automask = SAM2AutomaticMaskGenerator(image_model)
    return video_predictor, automask


@torch.no_grad()
def sam2_generate_init_masks(automask, first_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
    # automask expects RGB uint8 HxWx3
    anns = automask.generate(first_rgb)
    masks, scores = [], []
    for ann in anns:
        m = ann.get("segmentation", None)
        if m is None:
            continue
        s = ann.get("predicted_iou", None)
        if s is None:
            s = ann.get("stability_score", 1.0)
        masks.append(m.astype(bool))
        scores.append(float(s))
    return masks, scores


def sam2_add_objects_by_boxes(video_predictor, state, frame_idx: int, boxes_xyxy: List[np.ndarray]) -> List[int]:
    """
    Robustly call SAM2 video predictor's add_new_points_or_box using keyword args only.
    Handles minor signature differences across SAM2 repo versions.
    """
    object_ids_final = []

    for obj_id, box in enumerate(boxes_xyxy, start=1):
        if box is None:
            continue
        box = np.asarray(box, dtype=np.float32)

        # Try a few common keyword signatures
        tried = []
        out = None

        # 1) Most common: (state=, frame_idx=, obj_id=, box=)
        try:
            out = video_predictor.add_new_points_or_box(
                state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
            )
        except TypeError as e:
            tried.append(("obj_id/box", str(e)))

        # 2) Some versions use object_id instead of obj_id
        if out is None:
            try:
                out = video_predictor.add_new_points_or_box(
                    state=state,
                    frame_idx=frame_idx,
                    object_id=obj_id,
                    box=box,
                )
            except TypeError as e:
                tried.append(("object_id/box", str(e)))

        # 3) Some versions want box as shape (1,4)
        if out is None:
            try:
                out = video_predictor.add_new_points_or_box(
                    state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=box.reshape(1, 4),
                )
            except TypeError as e:
                tried.append(("obj_id/box(1,4)", str(e)))

        if out is None:
            msg = "add_new_points_or_box signature mismatch. Tried:\n"
            msg += "\n".join([f"  - {name}: {err}" for name, err in tried])
            raise TypeError(msg)

        # out should be (frame_idx, object_ids, masks)
        _, object_ids, _ = out
        object_ids_final = [int(x) for x in object_ids]

    return object_ids_final


# =========================
# Main
# =========================
@torch.no_grad()
def main(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)

    seq2frames = group_frames_by_seq(cfg.image_dir)
    if not seq2frames:
        raise FileNotFoundError(f"No seqX_frameXXXXX images found in: {cfg.image_dir}")

    print(f"[Info] Found {len(seq2frames)} sequences in {cfg.image_dir}")
    print(f"[Info] Device={cfg.device}, use_fp16={cfg.use_fp16}")
    print(f"[Info] SAM2 ckpt={cfg.sam2_ckpt}")
    print(f"[Info] SAM2 cfg ={cfg.sam2_config}")

    # Bank path
    bank_path = cfg.bank_path.strip()
    if not bank_path:
        bank_path = os.path.join(cfg.out_dir, "proto_bank.npy")

    bank: Optional[PrototypeBank] = None
    viz_dir = os.path.join(cfg.out_dir, "_viz") if cfg.viz else ""

    # Build SAM2 predictors
    video_predictor, automask = build_sam2_predictors(cfg)

    # Autocast context
    use_autocast = cfg.use_fp16 and cfg.device.startswith("cuda")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else torch.no_grad()
    )

    for seq_name in tqdm(sorted(seq2frames.keys()), desc="Processing sequences"):
        frame_paths = seq2frames[seq_name]
        if not frame_paths:
            continue

        # Load full video into memory (list of RGB frames)
        video_frames_rgb: List[np.ndarray] = []
        for p in frame_paths:
            rgb = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
            video_frames_rgb.append(rgb)

        H, W = video_frames_rgb[0].shape[:2]
        first_rgb = video_frames_rgb[0]

        with torch.inference_mode(), autocast_ctx:
            # First-frame auto masks
            init_masks, init_scores = sam2_generate_init_masks(automask, first_rgb)

            # Filter by area
            filt_masks, filt_scores = [], []
            for m, s in zip(init_masks, init_scores):
                if mask_area(m) >= cfg.init_min_area:
                    filt_masks.append(m)
                    filt_scores.append(float(s))

            if not filt_masks:
                print(f"[Warn] {seq_name}: no init masks after filtering, skip.")
                continue

            # NMS
            keep_idx = nms_masks(filt_masks, filt_scores, cfg.init_nms_iou)
            keep_idx = keep_idx[: cfg.init_max_masks]
            init_masks = [filt_masks[i] for i in keep_idx]
            init_scores = [filt_scores[i] for i in keep_idx]

            # Convert init masks -> boxes (xyxy)
            init_boxes = []

            for m in init_masks:
                box = _mask_to_xyxy(m)
                if box is None:
                    continue
                init_boxes.append(box.astype(np.float32))

            if len(init_boxes) == 0:
                print(f"[Warn] {seq_name}: no valid init boxes, skip.")
                continue

            for m in init_masks:
                box = _mask_to_xyxy(m)
                if box is not None:
                    init_boxes.append(box)

            if not init_boxes:
                print(f"[Warn] {seq_name}: no valid boxes from init masks, skip.")
                continue

            # Init tracking state (SAM2 example uses predictor.init_state(<your_video>))
            jpeg_dir, orig_paths = prepare_jpeg_sequence_from_png(frame_paths, quality=95)
            try:
                state = video_predictor.init_state(jpeg_dir)

                # Add objects on frame 0 with box prompts
                _ = sam2_add_objects_by_boxes(video_predictor, state, frame_idx=0, boxes_xyxy=init_boxes)

                # Containers
                # per_frame_outputs[frame_idx] = (tids, masks_list, scores_list)
                per_frame_outputs: Dict[int, Tuple[List[int], List[np.ndarray], List[float]]] = {}

                # Track-level embedding accumulators
                track_embed: Dict[int, torch.Tensor] = {}
                track_count: Dict[int, int] = {}

                def update_track_embed(tid: int, mask_bool: np.ndarray):
                    e = mask_geom_embedding(mask_bool, H, W)
                    et = torch.from_numpy(e).to(cfg.device, dtype=torch.float32)
                    if tid not in track_embed:
                        track_embed[tid] = et
                        track_count[tid] = 1
                    else:
                        c = track_count[tid]
                        track_embed[tid] = (track_embed[tid] * c + et) / float(c + 1)
                        track_count[tid] = c + 1

                # Propagate across video
                for frame_idx, object_ids, masks in video_predictor.propagate_in_video(state):
                    # masks often: torch.Tensor [N,H,W] float/logits/bool
                    if isinstance(masks, torch.Tensor):
                        masks_np = masks.detach().cpu().numpy()
                    else:
                        masks_np = np.asarray(masks)

                    tids = [int(x) for x in object_ids]
                    masks_list: List[np.ndarray] = []
                    scores_list: List[float] = []

                    # propagate often doesn't provide per-object scores; use 1.0 as placeholder
                    for k in range(masks_np.shape[0]):
                        m = masks_np[k]
                        m_bool = (m > 0).astype(bool)
                        if mask_area(m_bool) < cfg.track_min_area:
                            continue
                        masks_list.append(m_bool)
                        scores_list.append(1.0)

                    # Keep tids aligned with masks_list:
                    # Some objects might have been filtered out. Filter tids accordingly by re-checking masks.
                    # Easiest: rebuild tids by iterating again with same filter.
                    tids_f = []
                    idx_keep = 0
                    for k in range(masks_np.shape[0]):
                        m_bool = (masks_np[k] > 0).astype(bool)
                        if mask_area(m_bool) < cfg.track_min_area:
                            continue
                        tids_f.append(tids[k])
                        update_track_embed(tids[k], m_bool)
                        idx_keep += 1

                    per_frame_outputs[int(frame_idx)] = (tids_f, masks_list, scores_list)
            finally:
                shutil.rmtree(jpeg_dir, ignore_errors=True)

            if not track_embed:
                print(f"[Warn] {seq_name}: tracker produced no masks after filtering, skip.")
                continue

            # Init/load PrototypeBank (dim known now)
            emb_dim = int(next(iter(track_embed.values())).numel())
            if bank is None:
                if cfg.reuse_bank:
                    if not os.path.isfile(bank_path):
                        raise FileNotFoundError(f"--reuse_bank true but bank not found: {bank_path}")
                    bank = PrototypeBank.load(bank_path, device=cfg.device)
                    if bank.dim != emb_dim:
                        raise RuntimeError(f"Loaded bank dim={bank.dim} mismatch embed dim={emb_dim}.")
                    print(f"[Info] Loaded PrototypeBank: {bank_path}, K={bank.K}, dim={bank.dim}")
                else:
                    bank = PrototypeBank(
                        dim=emb_dim,
                        sim_thresh=cfg.proto_sim_thresh,
                        ema=cfg.proto_ema,
                        max_k=cfg.proto_max,
                        device=cfg.device,
                    )
                    print(f"[Info] Init PrototypeBank: dim={emb_dim}")

            # Map track_id -> global_id (cross-seq)
            track_to_global: Dict[int, int] = {}
            for tid, et in track_embed.items():
                gid, _ = bank.match_or_create(et)
                track_to_global[int(tid)] = int(gid)

            # Save per-frame label PNGs
            max_gid = bank.K - 1
            for local_i, img_path in enumerate(orig_paths):
                img_path = orig_paths[local_i]
                base = os.path.splitext(os.path.basename(img_path))[0]  # seq2_frame00001
                save_path = os.path.join(cfg.out_dir, base + ".png")

                if local_i in per_frame_outputs:
                    tids, masks_list, scores_list = per_frame_outputs[local_i]
                    gids = [track_to_global.get(int(t), -1) for t in tids]
                    label_hw = write_masks_to_label_map(
                        H, W, masks_list, gids, scores_list, min_area=cfg.track_min_area
                    )
                else:
                    label_hw = np.full((H, W), -1, dtype=np.int32)

                save_full_label_png(cfg, save_path, label_hw, max_label_id=max_gid)

                if cfg.viz:
                    viz_path = os.path.join(viz_dir, base + ".png")
                    img = Image.open(img_path).convert("RGB")
                    save_tiny_viz(cfg, img, label_hw, viz_path)

    # Save bank
    if bank is not None:
        bank.save(bank_path)
        print(f"[Info] PrototypeBank saved: {bank_path}, K={bank.K}, dim={bank.dim}")

    print(f"[Done] Labels saved to: {cfg.out_dir}")


if __name__ == "__main__":
    CFG = parse_args_to_cfg()

    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.seed)

    main(CFG)
