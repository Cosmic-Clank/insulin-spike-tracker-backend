import base64
import os


def save_base64_images(images: list[str], folder: str = "images") -> None:
    os.makedirs(folder, exist_ok=True)
    for idx, image in enumerate(images, start=1):
        image_data = base64.b64decode(image.split(",")[-1])
        with open(f"{folder}/{idx}.jpg", "wb") as f:
            f.write(image_data)
