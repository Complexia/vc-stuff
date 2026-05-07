# Vehicle Colour Classifier

This crate exposes the requested CLI:

```sh
cargo run --release -- --manifest ../boundingBoxes.json --images ../images --out predictions.json
```

or, once installed:

```sh
vehicle-colour --manifest boundingBoxes.json --images ./images --out predictions.json
```

This second take adds a hybrid segmentation-plus-colour pipeline. If you have
AI-generated masks, pass them in:

```sh
cargo run --release -- \
  --manifest ../boundingBoxes.json \
  --images ../images \
  --masks masks \
  --out predictions.json
```

## Approach

The classifier keeps colour estimation pixel-based, but first narrows sampling
to a segmentation mask. That is the important production improvement: colour is
still a physical property of pixels, while the ML model handles the hard visual
task of isolating the vehicle from the road, sky, windows, tyres, and other
vehicles.

For each bounding box it:

1. Loads the image and clamps the box to image bounds.
2. Loads an optional AI mask for that detection from `--masks`.
3. Falls back to a conservative bbox-prior mask if no mask is available, so the CLI remains runnable without model weights.
4. Samples only masked pixels.
5. Applies a simple gray-world white balance so colour-graded frames do not turn neutral cars blue/orange.
6. Filters out likely glare, deep shadows, glass/tyres, lane markings, plates, and lamps where enough cleaner pixels are available.
7. Uses weighted palette voting plus an aggregate RGB/HSV fallback.
8. Snaps the result to the fixed palette:

```text
black, white, gray, silver, red, orange, yellow, green, blue, purple, brown
```

## Mask Format

The Rust CLI accepts PNG masks in `--masks`. White pixels are vehicle, black
pixels are ignored. It checks these names in order:

```text
{global_detection_index}.png
{image_stem}__{global_detection_index}.png
{image_stem}__{per_image_detection_index}.png
{image_stem}.png
```

Per-detection masks are expected to be crop-sized and aligned to the bounding
box. `{image_stem}.png` masks are expected to be full-image masks.

## Optional SAM Helper

`scripts/sam_masks.py` can generate per-detection masks with Meta Segment
Anything if you provide a SAM checkpoint:

```sh
python scripts/sam_masks.py \
  --manifest ../boundingBoxes.json \
  --images ../images \
  --checkpoint sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --out masks
```

Then run the Rust classifier with `--masks masks`.

The hard cases are neutral cars under strong lighting, where "gray" vs
"silver" is subjective; the code treats brighter, low-saturation neutrals as
silver and darker neutrals as gray or black.

Take 5 adds a narrow underexposed-neutral correction after SAM masking: if the
classifier would return black for a low-saturation vehicle, it checks the
brighter neutral masked pixels first. If those upper-percentile pixels are
clearly gray/silver, it returns gray or silver instead of treating the whole
dark frame as black.
