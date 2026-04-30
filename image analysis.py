from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model once
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def analyze_image(image):
    """
    Generate caption from image
    """
    inputs = processor(image, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_length=30,
        num_beams=4
    )

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption
