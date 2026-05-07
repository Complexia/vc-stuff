# Take 3 Spec

Pipeline:

```text
provided bbox -> local SAM mask -> masked crop PNG -> OpenRouter Qwen3-VL -> palette colour
```

The Rust CLI no longer decides colour from pixel averages. It creates a masked
crop for each detection and asks `qwen/qwen3-vl-30b-a3b-instruct` to choose one
palette colour while ignoring shadows, windows, tyres, reflections, road, and
the white masked background.

The OpenRouter API key can be supplied with `--openrouter-api-key` or via
`OPENROUTER_API_KEY` in `.env`. The model can be supplied with `--model`, via
`MODEL` in `.env`, or falls back to `qwen/qwen3-vl-30b-a3b-instruct`.


# Vehicle Colour Classifier, Take 3

This version uses the hybrid VLM pipeline:

```text
bbox crop -> SAM mask -> masked crop image -> Qwen3-VL prompt through OpenRouter
```

The bounding boxes still come from `boundingBoxes.json`. SAM runs locally to
isolate vehicle pixels. The masked crop is then sent to OpenRouter using
`qwen/qwen3-vl-30b-a3b-instruct` by default.

## API Key

Pass your OpenRouter key directly:

```sh
cargo run --release -- \
  --manifest ../boundingBoxes.json \
  --images ../images \
  --masks masks \
  --out predictions.json \
  --openrouter-api-key sk-or-...
```

Or create `.env` in this folder:

```sh
OPENROUTER_API_KEY=sk-or-...
MODEL=qwen/qwen3-vl-30b-a3b-instruct
```

The CLI reads `.env` automatically. `--openrouter-api-key` and `--model`
override environment values when provided.

## End-To-End Run

From this folder:

```sh
./run.sh
```

`run.sh` will:

1. Create `.venv` if needed.
2. Install Python dependencies for SAM.
3. Download the SAM ViT-B checkpoint if needed.
4. Generate masks into `masks/`.
5. Run the Rust classifier with `--masks`.
6. Write `predictions.json`.

You can override paths with environment variables, and set the model via
`MODEL` in `.env` or the shell:

```sh
MANIFEST=../boundingBoxes.json \
IMAGES=../images \
OUT=predictions.json \
MODEL=qwen/qwen3-vl-30b-a3b-instruct \
./run.sh
```

## What It Does

1. Uses the provided bounding box for each detection.
2. Runs SAM locally with that box as a prompt.
3. Builds a crop where masked vehicle pixels are preserved and non-vehicle
   background is painted white.
4. Sends that masked crop to Qwen3-VL through OpenRouter with a constrained
   prompt asking for exactly one palette colour.
5. Normalizes the model response to:

```text
black, white, gray, silver, red, orange, yellow, green, blue, purple, brown
```
