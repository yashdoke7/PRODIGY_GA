from PIL import Image
import os

# Paths
input_img_path = './pix2pix_inputs/facade_input_1.png'
output_img_path = './pix2pix_result.png'  # Save this from Streamlit UI

# Load images
input_img = Image.open(input_img_path)
output_img = Image.open(output_img_path)

# Create side-by-side comparison
width = input_img.width + output_img.width + 20  # 20px gap
height = max(input_img.height, output_img.height)

comparison = Image.new('RGB', (width, height), (255, 255, 255))

# Paste images
comparison.paste(input_img, (0, 0))
comparison.paste(output_img, (input_img.width + 20, 0))

# Save
comparison.save('./linkedin_pix2pix_demo.png')
print("✓ Saved comparison to linkedin_pix2pix_demo.png")
