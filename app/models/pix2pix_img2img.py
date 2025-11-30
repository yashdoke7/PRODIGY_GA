import torch
from PIL import Image
import torchvision.transforms as transforms

class Pix2Pix:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if self.model is None:
            print("Loading Pix2Pix model from PyTorch Hub...")
            try:
                # Try to load pretrained model
                self.model = torch.hub.load(
                    'junyanz/pytorch-CycleGAN-and-pix2pix', 
                    'pix2pix',
                    model_name='edges2shoes',  # or 'facades'
                    pretrained=True,
                    map_location=self.device,
                    verbose=False
                )
                self.model.eval()
                print("Pix2Pix model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load Pix2Pix model: {e}")
                print("Using edge detection fallback")
                self.model = "edge_detection"  # Fallback to simple edge detection

    def preprocess_img(self, image):
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img_tensor = transform(image).unsqueeze(0)
        return img_tensor.to(self.device)

    def postprocess_img(self, tensor):
        tensor = tensor.squeeze(0).cpu().detach()
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
        return transforms.ToPILImage()(tensor)

    def translate(self, input_img: Image.Image):
        self.load_model()
        
        # If model loading failed, use simple edge detection
        if self.model == "edge_detection":
            import cv2
            import numpy as np
            
            # Convert PIL to CV2
            img_array = np.array(input_img.convert('RGB'))
            
            # Apply Canny edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Convert back to PIL
            return Image.fromarray(edges).convert('RGB')
        
        # Use actual model
        inp = self.preprocess_img(input_img)
        with torch.no_grad():
            out = self.model(inp)
        return self.postprocess_img(out)
