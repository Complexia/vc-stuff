# Take-Home: Vehicle Colour Classifier

## The Problem

You're given car images with bounding boxes around each vehicle. For each detection, output the car's colour from this fixed palette:

```
black, white, gray, silver, red, orange, yellow, green, blue, purple, brown
```

The approach is up to you. We're gauging thought process, creativity and decision making. We'll have a follow-up call to walk through your code together.


## Input / Output

The input json `boundingBoxes.json` is fairly self-explanatory.

```json
[
  {"image": "car_0.jpg", "bbox_pixels": {"left": 134.06, "top": 223.12, "width": 1651.88, "height": 643.12}},
  ...
]
```

Output (one entry per input detection) e.g.:

```json
[
  {"image": "car_0.jpg", "bbox_pixels": {...}, "colour": "silver"},
  ...
]
```

## Deliverables

- A Rust crate that exposes a CLI:

  ```
  vehicle-colour --manifest boundingBoxes.json --images ./images --out predictions.json
  ```

- If you like, a **brief** README.md explaining your approach
