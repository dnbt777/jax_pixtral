###
### run_lora_inference.py
### --- runs inference on a single example prompt (context)
###

from jax_pixtral.inference import get_completion



# create prompt with images (context)
prompt = "For each image, write one sentence describing its contents." 
chess_image = "images/chess.png"
text_image = "images/gqa.png"
highres_image = "images/bed.jpg"
context = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": { "url": chess_image }
            },
            {
                "type": "image_url",
                "image_url": { "url": text_image }
            },
            {
                "type": "image_url",
                "image_url": { "url": highres_image }
            },
            {
                "type": "text",
                "text": prompt
            },
        ],
    },
]



# run inference on prompt
completion = get_completion(context, max_tokens=256, temp=0.0)

print(completion)
