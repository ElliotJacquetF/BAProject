import os
from PIL import Image
from PIL import ImageDraw
import json

# Set the input and output directories
input_dir = "/home/elliot/cours/BA6/BAProject/annotations/placid"
output_dir = "/home/elliot/cours/BA6/BAProject/annotations/output"

# Load the annotations
with open("/home/elliot/cours/BA6/BAProject/annotations/instances_default.json", "r") as f:
    data = json.load(f)

# Define the get_image_id function
def get_image_id(filename, data):
    # Remove ".jpg1.borderless" from file name
    file_name = os.path.basename(filename)
    file_name = file_name.replace(".jpg1.borderless", "")

    # Find the image ID in the data
    for img in data["images"]:
        if img["file_name"] == "placid/" + file_name:
            return img["id"]

    # Return None if no matching image is found
    return None

# Loop over all image files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Load the image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)

        image_id = get_image_id(filename, data)

        # Filter annotations for this image
        annotations = [anno for anno in data["annotations"] if anno["image_id"] == image_id]

        # Get the segmentation data for each annotation object
        segmentations = []
        for anno in annotations:
            if isinstance(anno.get("segmentation"), list) and len(anno["segmentation"]) > 0:
                segmentations.append(anno["segmentation"][0])

        # Draw the polygon on the image for each annotation
        draw = ImageDraw.Draw(image)
        for seg in segmentations:
            draw.polygon(seg, outline="red")

        # Save the annotated image
        output_path = os.path.join(output_dir, "a_" + filename)
        image.save(output_path)
