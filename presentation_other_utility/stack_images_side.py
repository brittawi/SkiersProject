from PIL import Image
import os

def combine_images_side_by_side(folder_path, output_path):
    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    image_files.sort()  # Optional: sort to keep a consistent order

    # Open all images
    images = [Image.open(os.path.join(folder_path, img)) for img in image_files]

    # Optionally resize images to the same height
    min_height = min(img.height for img in images)
    resized_images = [img.resize((int(img.width * min_height / img.height), min_height), Image.LANCZOS) for img in images]

    # Calculate total width
    total_width = sum(img.width for img in resized_images)

    # Create a new blank image
    combined_image = Image.new('RGB', (total_width, min_height))

    # Paste images side by side
    x_offset = 0
    for img in resized_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the output
    combined_image.save(output_path)
    print(f"Saved combined image at: {output_path}")

if __name__ == "__main__":
    folder = "/Users/emillundin/Desktop/G3Front"  # Folder containing your images
    output = "combined_output.jpg"  # Output file path
    combine_images_side_by_side(folder, output)
