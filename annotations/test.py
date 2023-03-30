import os
from PIL import Image
from PIL import ImageDraw
import json
import matplotlib.path as mplPath

def get_annotation_id(image, x, y):
    # Get the annotations for the image
    with open("/path/to/annotations.json", "r") as f:
        data = json.load(f)
    annotations = [anno for anno in data["annotations"] if anno["image_id"] == image.id]

    # Check if the point (x, y) is in any of the annotations
    for anno in annotations:
        if anno["segmentation"]:
            seg = anno["segmentation"][0]
            path = mplPath(seg)
            if path.contains_point((x, y)):
                return anno["id"]

    # If the point isn't in any annotation, return -1
    return -1
