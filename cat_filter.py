from PIL import Image

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_cat_probability(photo_path):
    image = Image.open(photo_path)
    inputs = processor(
        text=["a photo with at least one cat", "a photo with no cat"],
        images=image,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities

    return round(probs[0][0].item() * 100, 2)
