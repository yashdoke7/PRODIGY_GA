import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

class StyleTransfer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg19(pretrained=True).features.to(self.device).eval()
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    def get_features(self, image, model):
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content layer
            '28': 'conv5_1'
        }
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features
    
    def gram_matrix(self, tensor):
        b, c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram
    
    def preprocess_img(self, image):
        # Resize to reasonable size to avoid memory issues
        max_size = 512
        if max(image.size) > max_size:
            scale = max_size / max(image.size)
            new_size = tuple(int(dim * scale) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def deprocess_img(self, tensor):
        image = tensor.cpu().clone().detach()
        image = image.squeeze(0)
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.clamp(0, 1)
        
        transform = transforms.ToPILImage()
        image = transform(image)
        return image
    
    def transfer(self, content_img: Image.Image, style_img: Image.Image, 
                 content_weight=1e5, style_weight=1e10, num_steps=300):
        
        # Convert images to RGB if needed
        if content_img.mode != 'RGB':
            content_img = content_img.convert('RGB')
        if style_img.mode != 'RGB':
            style_img = style_img.convert('RGB')
        
        # Preprocess images
        content = self.preprocess_img(content_img)
        style = self.preprocess_img(style_img)
        
        # Resize style to match content
        if style.size() != content.size():
            style = torch.nn.functional.interpolate(
                style, size=content.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Get features
        content_features = self.get_features(content, self.model)
        style_features = self.get_features(style, self.model)
        
        # Calculate style grams
        style_grams = {layer: self.gram_matrix(style_features[layer]) 
                      for layer in style_features}
        
        # Initialize target image with content
        target = content.clone().requires_grad_(True)
        
        # Use Adam optimizer (more stable than LBFGS)
        optimizer = optim.Adam([target], lr=0.003)
        
        # Style layers and weights
        style_weights = {
            'conv1_1': 1.0,
            'conv2_1': 0.8,
            'conv3_1': 0.5,
            'conv4_1': 0.3,
            'conv5_1': 0.1
        }
        
        print(f"Starting style transfer for {num_steps} steps...")
        
        for step in range(num_steps):
            # Get features from target
            target_features = self.get_features(target, self.model)
            
            # Content loss
            content_loss = torch.mean(
                (target_features['conv4_2'] - content_features['conv4_2']) ** 2
            )
            
            # Style loss
            style_loss = 0
            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                style_gram = style_grams[layer]
                layer_style_loss = style_weights[layer] * torch.mean(
                    (target_gram - style_gram) ** 2
                )
                b, c, h, w = target_feature.shape
                style_loss += layer_style_loss / (c * h * w)
            
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Print progress
            if (step + 1) % 50 == 0:
                print(f"Step {step + 1}/{num_steps}, Loss: {total_loss.item():.4f}")
        
        print("Style transfer complete!")
        return self.deprocess_img(target)
