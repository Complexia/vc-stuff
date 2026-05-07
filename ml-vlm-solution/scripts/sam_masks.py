#!/usr/bin/env python3
"""Generate per-detection vehicle masks with Segment Anything.

This helper is intentionally outside the Rust binary because SAM model weights
are large and deployment-specific. It writes crop-sized PNG masks that the Rust
CLI can consume with `--masks`.

Example:
    python scripts/sam_masks.py \
      --manifest ../boundingBoxes.json \
      --images ../images \
      --checkpoint sam_vit_h_4b8939.pth \
      --model-type vit_h \
      --out masks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-type", default="vit_h")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    detections = json.loads(args.manifest.read_text())
    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    current_image_name = None
    image_rgb = None

    for index, detection in enumerate(detections):
        image_name = detection["image"]
        if image_name != current_image_name:
            image_bgr = cv2.imread(str(args.images / image_name), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(args.images / image_name)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            current_image_name = image_name

        bbox = detection["bbox_pixels"]
        left = max(0, int(np.floor(bbox["left"])))
        top = max(0, int(np.floor(bbox["top"])))
        right = min(image_rgb.shape[1], int(np.ceil(bbox["left"] + bbox["width"])))
        bottom = min(image_rgb.shape[0], int(np.ceil(bbox["top"] + bbox["height"])))
        box = np.array([left, top, right, bottom])

        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        mask = masks[int(np.argmax(scores))]
        crop_mask = (mask[top:bottom, left:right] * 255).astype(np.uint8)
        cv2.imwrite(str(args.out / f"{index}.png"), crop_mask)


if __name__ == "__main__":
    main()
