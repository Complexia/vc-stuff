There are 3 solutions here

1)
no-ml-solution simply uses the bbox for the pixels, ignores the edges (more chance to avoid road, sky, etc), converts pixels from RGB to HSV, and gets color average. It's mostly accurate but struggles on black and white images or images taken in darker settings (like car_1.jpg)

2) 
ml-solution takes the bbox pixels, runs it through SAM (ran locally) to get the exact pixels of the vehicles (creates the masks), then does the same thing no-ml-soliton does. It can be slightly more accurate as there is less chance pixels from foreign objects that are not vehicle (like the road, sky, etc) get mixed up in the calculation. It has no effect on the accuracy of darker or black and white images like car_1.jpg and can still get it wrong. 

3)
ml-vlm-solution takes the bbox pixels, runs it through SAM for masks, then sends those as the prompt to a VLM (or any LLM with visual capability) via OpenRouter. VLM returns the color of the vehicle as semantically inferred. qwen3-vl struggled on car_01.jpg and got it wrong, and gpt-5.4-mini got it right. Can be tested with any visual model by adjusting the MODEL variable in .env
(note, feel free to use my OPENROUTER api key for testing, I put a $10 limit on it)

4)
no-ml-improved is basically an improved version of no-ml, with the following difference:

adds a narrow underexposed-neutral correction: if the classifier would
return black for a low-saturation vehicle, it checks the brighter neutral body
pixels first. If those upper-percentile pixels are clearly gray/silver, it
returns gray or silver instead of treating the whole dark frame as black.

5) ml-improved is the same as 4 but with SAM

Note, 4 and 5 correctly predict car_01.jpg as well as all of the other cars with no VLMs. 