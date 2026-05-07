How to run: ./run.sh 
make sure you have your .env set correctly with openrouter api key and model



## What It Does

1. Uses the provided bounding box for each detection.
2. Runs SAM locally with that box as a prompt.
3. Builds a crop where masked vehicle pixels are preserved and non-vehicle
   background is painted white.
4. Sends that masked crop to a visual model through OpenRouter with a constrained
   prompt asking for exactly one palette colour.
5. Normalizes the model response to: black, white, gray, silver, red, orange, yellow, green, blue, purple, brown

