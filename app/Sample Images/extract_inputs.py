from PIL import Image
import os

# Process validation images (best quality)
val_dir = './facades/val'
output_dir = './pix2pix_inputs'
os.makedirs(output_dir, exist_ok=True)

# Get all images
files = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]

print(f"Found {len(files)} images")

# Extract first 5 for demo
for i, filename in enumerate(files[:5]):
    img_path = os.path.join(val_dir, filename)
    img = Image.open(img_path)
    
    # Images are 512x256 (input on left, output on right)
    width, height = img.size
    
    # Split in half - left side is the segmentation input
    input_img = img.crop((0, 0, width // 2, height))
    
    # Save
    output_path = os.path.join(output_dir, f'facade_input_{i+1}.png')
    input_img.save(output_path)
    print(f"✓ Saved {output_path}")

print(f"\n✓ Done! Input images saved to {output_dir}/")
