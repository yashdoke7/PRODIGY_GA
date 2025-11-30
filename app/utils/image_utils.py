import os
from PIL import Image

def save_image(image: Image.Image, out_path):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    image.save(out_path)
    return out_path
