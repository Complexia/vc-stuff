# Vehicle Colour Classifier

This crate exposes the requested CLI:

```sh
cargo run --release -- --manifest ../boundingBoxes.json --images ../images --out predictions.json
```

or, once installed:

```sh
vehicle-colour --manifest boundingBoxes.json --images ./images --out predictions.json
```

## Approach

The classifier is deliberately simple and explainable. For each bounding box it:

1. Loads the image and clamps the box to image bounds.
2. Samples pixels from the central body area of the detection, down-weighting the top/bottom edges where sky, road, windows, tyres, and bumpers are common.
3. Applies a simple gray-world white balance so colour-graded frames do not turn neutral cars blue/orange.
4. Filters out likely glare, deep shadows, glass/tyres, lane markings, plates, and lamps where enough cleaner pixels are available.
5. Uses weighted palette voting plus an aggregate RGB/HSV fallback.
6. Snaps the result to the fixed palette:

```text
black, white, gray, silver, red, orange, yellow, green, blue, purple, brown
```

There is no training data here, so the implementation favours transparent heuristics over a model. The hard cases are neutral cars under strong lighting, where "gray" vs "silver" is subjective; the code treats brighter, low-saturation neutrals as silver and darker neutrals as gray or black.
