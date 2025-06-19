import json
import os
import re
import requests
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from pypdf import PdfReader

from ai_custom_utils.helper import (
    get_llama_api_key,
    get_together_api_key,
    get_llama_base_url,
)
from openai import OpenAI
from together import Together


def llama4(
    user_prompt,
    system_prompt="You are a helpful assistant",
    image_urls=[],
    model="Llama-4-Scout-17B-16E-Instruct-FP8",
    debug=False,
):  # Llama-4-Maverick-17B-128E-Instruct-FP8
    image_urls_content = []
    for url in image_urls:
        image_urls_content.append({"type": "image_url", "image_url": {"url": url}})

    content = [{"type": "text", "text": user_prompt}]
    content.extend(image_urls_content)

    client = OpenAI(api_key=get_llama_api_key(), base_url=get_llama_base_url())

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )

    if debug:
        print(response)

    return response.choices[0].message.content


def llama4_together(
    prompt,
    image_urls=[],
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    debug=False,
):
    image_urls_content = []
    for url in image_urls:
        image_urls_content.append({"type": "image_url", "image_url": {"url": url}})

    content = [{"type": "text", "text": prompt}]
    content.extend(image_urls_content)

    client = Together(api_key=get_together_api_key())
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}], temperature=0
    )

    if debug:
        print(response)

    return response.choices[0].message.content


def pdf2text(file: str):
    text = ""
    with Path(file).open("rb") as f:
        reader = PdfReader(f)
        text = "\n\n".join([page.extract_text() for page in reader.pages])

    return text


with open("files/pr.json", "r") as f:
    all_pr_content = json.load(f)


def get_pr_content(repo_owner, repo_name, pr_number, token=None):
    if str(pr_number) not in all_pr_content:
        return None
    return all_pr_content[str(pr_number)]


def get_pull_requests(repo_owner, repo_name, token=None):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"

    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    pull_requests = []
    params = {}
    params["state"] = "all"
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = json.loads(response.text)

        pull_requests.extend(data)

        while "next" in response.links:
            next_url = response.links["next"]["url"]

            response = requests.get(next_url, headers=headers)

            if response.status_code == 200:
                data = json.loads(response.text)

                pull_requests.extend(data)
            else:
                print(f"Failed to retrieve next page: {response.text}")
                break
    else:
        print(f"Failed to retrieve pull requests: {response.text}")

    return pull_requests


def get_pr_content_live(repo_owner, repo_name, pr_number, token=None):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = json.loads(response.text)
        content = data["title"] + (data["body"] if data["body"] else "")
        return content
    else:
        print(f"Failed to retrieve pull request content: {response.text}")
        return None


def get_issues(repo_owner, repo_name, token=None):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"

    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    issues = []
    params = {
        "state": "all",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = json.loads(response.text)

        issues.extend([issue for issue in data if "pull_request" not in issue])

        while "next" in response.links:
            next_url = response.links["next"]["url"]

            response = requests.get(next_url, headers=headers)

            if response.status_code == 200:
                data = json.loads(response.text)
                issues.extend([issue for issue in data if "pull_request" not in issue])
            else:
                print(f"Failed to retrieve next page: {response.text}")
                break
    else:
        print(f"Failed to retrieve pull requests: {response.text}")

    return issues


def download_and_extract_repo(repo_url, extract_dir):
    """Download and extract the GitHub repository ZIP file."""
    if repo_url.endswith("/"):
        repo_url = repo_url[:-1]
    zip_url = f"{repo_url}/archive/refs/heads/main.zip"

    response = requests.get(zip_url)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(extract_dir)


def get_py_files(repo_dir):
    """Get a list of Python files in the repository."""
    py_files = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            _, file_extension = os.path.splitext(file_path)
            if file_extension == ".py":
                py_files.append(file_path)
    return py_files


def write_files_to_text(py_files, output_file):
    """Write the paths and contents of non-binary files to a text file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for file_path in py_files:
            print(f"Writing {file_path}")
            f.write(f"Path: {file_path}\n")
            f.write("Content:\n")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                f.write(file.read())
            f.write("\n\n")


def plot_tiled_image(width, height, tile_size, patch_size):
    # Create a new image
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Divide the image into tiles
    for i in range(0, width, tile_size):
        for j in range(0, height, tile_size):
            # Draw a rectangle for each tile
            draw.rectangle((i, j, i + tile_size, j + tile_size), outline="black")

            # Divide each tile into patches
            for x in range(i, i + tile_size, patch_size):
                for y in range(j, j + tile_size, patch_size):
                    # Draw a rectangle for each patch
                    draw.rectangle(
                        (x, y, x + patch_size, y + patch_size), outline="gray"
                    )

    # Add separator lines with text
    font = ImageFont.load_default()
    font = font.font_variant(size=28)

    for i in range(tile_size, width, tile_size):
        draw.line((i, 0, i, height), fill="black")
        draw.text((i - 150, height // 5), "<tile_x_separator|>", font=font, fill="blue")
        draw.text(
            (i - 150, height // 1.5), "<tile_x_separator|>", font=font, fill="blue"
        )
    for j in range(tile_size, height, tile_size):
        draw.line((0, j, width, j), fill="black")
        draw.text((width // 2, j - 20), "<tile_y_separator|>", font=font, fill="blue")
    draw.line((0, height - 10, width, height - 10), fill="black")
    draw.text((width // 2, height - 40), "<tile_y_separator|>", font=font, fill="blue")

    # Add additional texts
    draw.text((10, 10), "Image Size: 768x768", font=font, fill="black")
    draw.text((10, 40), "Tile Size: 336x336; # of Tiles: 9", font=font, fill="black")
    draw.text(
        (10, 70),
        "Patch Size: 28x28; # of Patches per Tile: 144 (12x12)",
        font=font,
        fill="black",
    )

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Display the image using matplotlib
    plt.imshow(img_array)
    plt.axis("off")  # Turn off the axis
    plt.show()


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
    bboxes = re.findall(r"<BBOX>(.*?)</BBOX>", output)

    # Initialize an empty list to store the tools
    tools = []

    # Split the output into lines
    lines = output.split("\n")

    # Iterate over the lines
    for line in lines:
        # Check if the line contains a tool name
        if "**" in line:
            # Extract the tool name
            name = line.strip().replace("*", "").strip()

            # Find the corresponding bounding box
            bbox = bboxes.pop(0)

            # Split the bounding box into coordinates
            x1, y1, x2, y2 = map(float, bbox.split(","))

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
        rect = patches.Rectangle(
            (tool.bbox.x1 * width, tool.bbox.y1 * height),
            (tool.bbox.x2 - tool.bbox.x1) * width,
            (tool.bbox.y2 - tool.bbox.y1) * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        # Add the patch to the axis
        ax.add_patch(rect)

        # Annotate the tool
        ax.text(tool.bbox.x1 * width, tool.bbox.y1 * height, tool.name, color="red")

    # Set the limits of the axis to the size of the image
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    # Show the plot
    plt.show()


def display_local_image(image_path):
    img = Image.open(image_path)
    plt.figure(figsize=(5, 4), dpi=200)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def parse_json(input_string: str):
    """
    Attempts to parse the given string as JSON. If direct parsing fails,
    it tries to extract a JSON snippet from code blocks formatted as:
        ```json
        ... JSON content ...
        ```
    or any code block delimited by triple backticks and then parses that content.
    Parameters:
        input_string (str): The input string which may contain JSON.
    Returns:
        The parsed JSON object.
    Raises:
        ValueError: If parsing fails even after attempting to extract a JSON snippet.
    """
    # Try to parse the string directly.
    try:
        return json.loads(input_string)
    except json.JSONDecodeError as err:
        error = err  # Proceed to try extracting a JSON snippet.
    # Define patterns to search for a JSON code block.
    patterns = [
        re.compile(
            r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
        ),  # code block with "json" label
        re.compile(
            r"```(.*?)```", re.DOTALL
        ),  # any code block delimited by triple backticks
    ]

    # Attempt extraction using each pattern in order.
    for pattern in patterns:
        match = pattern.search(input_string)
        if match:
            json_candidate = match.group(1).strip()
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                # Continue trying if extraction from the code block didn't result in valid JSON.
                continue
    # If all attempts fail, raise an error.
    raise error


def evaluate(
    ground_truth: Any, predictions: Any, strict_json: bool = True
) -> Dict[str, Any]:
    result = {
        "is_valid_json": False,
        "correct_categories": 0.0,
        "correct_sentiment": False,
        "correct_urgency": False,
    }
    try:
        ground_truth = (
            ground_truth
            if isinstance(ground_truth, dict)
            else (json.loads(ground_truth) if strict_json else parse_json(ground_truth))
        )
        predictions = (
            predictions
            if isinstance(predictions, dict)
            else (json.loads(predictions) if strict_json else parse_json(predictions))
        )
    except (json.JSONDecodeError, ValueError):
        pass
    else:
        result["is_valid_json"] = True

        # Handle missing categories in predictions
        correct_categories = 0
        total_categories = len(ground_truth.get("categories", {}))

        if total_categories > 0 and "categories" in predictions:
            for k in ground_truth["categories"].keys():
                # Check if the category exists in predictions before comparing
                if k in predictions["categories"]:
                    if ground_truth["categories"][k] == predictions["categories"][k]:
                        correct_categories += 1
                # Missing category counts as incorrect

            result["correct_categories"] = correct_categories / total_categories

        # Handle missing sentiment and urgency fields
        result["correct_sentiment"] = predictions.get("sentiment") == ground_truth.get(
            "sentiment", None
        )
        result["correct_urgency"] = predictions.get("urgency") == ground_truth.get(
            "urgency", None
        )

    # Calculate total score
    correct_fields = [v for k, v in result.items() if k.startswith("correct_")]
    result["total"] = (
        sum(correct_fields) / len(correct_fields) if correct_fields else 0.0
    )

    return result
