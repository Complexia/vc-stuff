How to run: run ./run.sh

How does run work?
1. Create .venv if needed.
  2. Install Python dependencies.
  3. Download the SAM ViT-B checkpoint if needed.
  4. Generate masks into masks/.
  5. Run the Rust classifier with --masks.
  6. Write predictions.json.


What this does?
Uses the provided boundary box in the picture for each vehicle detection.
Runs it through SAM to detect the actual car pixels.
Converts those pixels from RGB to HSV (for hues and brightness)
Calculates the color from the pallete using pixel average.
Outputs car color. 

Take 5 adds a narrow underexposed-neutral correction after SAM masking: if the
classifier would return black for a low-saturation vehicle, it checks the
brighter neutral masked pixels first. If those upper-percentile pixels are
clearly gray/silver, it returns gray or silver instead of treating the whole
dark frame as black.
