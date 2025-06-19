import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from llama_api_client import LlamaAPIClient
from together import Together
from ai_custom_utils.helper import get_llama_api_key, get_together_api_key

def llama4(prompt, image_urls=[], model="Llama-4-Scout-17B-16E-Instruct-FP8"): # Llama-4-Maverick-17B-128E-Instruct-FP8
  image_urls_content = []
  for url in image_urls:
    image_urls_content.append({"type": "image_url", "image_url": {"url": url}})

  content = [{"type": "text", "text": prompt}]
  content.extend(image_urls_content)

  client = LlamaAPIClient(api_key=get_llama_api_key())

  response = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": content
    }],
    temperature=0
  )

  return response.completion_message.content.text


def llama4_together(prompt, image_urls=[], model="meta-llama/Llama-4-Scout-17B-16E-Instruct"):
  image_urls_content = []
  for url in image_urls:
    image_urls_content.append({"type": "image_url", "image_url": {"url": url}})

  content = [{"type": "text", "text": prompt}]
  content.extend(image_urls_content)

  client = Together(api_key = get_together_api_key())
  response = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": content
    }],
    temperature=0
  )

  return response.choices[0].message.content

  
def plot_tiled_image(width, height, tile_size, patch_size):
    # Create a new image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Divide the image into tiles
    for i in range(0, width, tile_size):
        for j in range(0, height, tile_size):
            # Draw a rectangle for each tile
            draw.rectangle((i, j, i + tile_size, j + tile_size),
                           outline='black')

            # Divide each tile into patches
            for x in range(i, i + tile_size, patch_size):
                for y in range(j, j + tile_size, patch_size):
                    # Draw a rectangle for each patch
                    draw.rectangle((x, y, x + patch_size, y + patch_size),
                                   outline='gray')

    # Add separator lines with text
    font = ImageFont.load_default()
    font = font.font_variant(size=28)

    for i in range(tile_size, width, tile_size):
        draw.line((i, 0, i, height), fill='black')
        draw.text((i - 150, height // 5), '<tile_x_separator|>', font=font,
                  fill='blue')
        draw.text((i - 150, height // 1.5), '<tile_x_separator|>', font=font,
                  fill='blue')
    for j in range(tile_size, height, tile_size):
        draw.line((0, j, width, j), fill='black')
        draw.text((width // 2, j - 20), '<tile_y_separator|>', font=font,
                  fill='blue')
    draw.line((0, height - 10, width, height - 10), fill='black')
    draw.text((width // 2, height - 40), '<tile_y_separator|>', font=font,
              fill='blue')


    # Add additional texts
    draw.text((10, 10), 'Image Size: 768x768', font=font, fill='black')
    draw.text((10, 40), 'Tile Size: 336x336; # of Tiles: 9', font=font,
              fill='black')
    draw.text((10, 70),
              'Patch Size: 28x28; # of Patches per Tile: 144 (12x12)',
              font=font, fill='black')

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Display the image using matplotlib
    plt.imshow(img_array)
    plt.axis('off')  # Turn off the axis
    plt.show()














import re
from pydantic import BaseModel
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define a model for the bounding box
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

# Define a model for the tool
class Tool(BaseModel):
    name: str
    bbox: BoundingBox

def parse_output(output: str) -> List[Tool]:
    # Use regular expressions to find all occurrences of <BBOX>...</BBOX>
    bboxes = re.findall(r'<BBOX>(.*?)</BBOX>', output)

    # Initialize an empty list to store the tools
    tools = []

    # Split the output into lines
    lines = output.split('\n')

    # Iterate over the lines
    for line in lines:
        # Check if the line contains a tool name
        if '**' in line:
            # Extract the tool name
            name = line.strip().replace('*', '').strip()

            # Find the corresponding bounding box
            bbox = bboxes.pop(0)

            # Split the bounding box into coordinates
            x1, y1, x2, y2 = map(float, bbox.split(','))

            # Create a Tool object and add it to the list
            tools.append(Tool(name=name, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)))

    return tools

def draw_bounding_boxes(img_path: str, tools: List[Tool]) -> None:
    # Open the image using PIL
    img = Image.open(img_path)

    # Get the width and height of the image
    width, height = img.size

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    # Iterate over the tools
    for tool in tools:
        # Create a rectangle patch
        rect = patches.Rectangle((tool.bbox.x1 * width, tool.bbox.y1 * height),
                                 (tool.bbox.x2 - tool.bbox.x1) * width,
                                 (tool.bbox.y2 - tool.bbox.y1) * height,
                                 linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the axis
        ax.add_patch(rect)

        # Annotate the tool
        ax.text(tool.bbox.x1 * width, tool.bbox.y1 * height, tool.name, color='red')

    # Set the limits of the axis to the size of the image
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    # Show the plot
    plt.show()


import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def display_local_image(image_path):
    img = Image.open(image_path)
    plt.figure(figsize=(5,4), dpi=200)
    plt.imshow(img)
    plt.axis('off')
    plt.show()