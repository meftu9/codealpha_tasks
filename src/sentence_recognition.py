"""
Sentence and Word Recognition Module.
Segments handwritten sentences into individual lines and characters,
classifies each character using the trained CNN model, detects word spaces,
and reconstructs the full text sentence.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import scipy.ndimage as ndi
import tensorflow as tf

from src.preprocessing import preprocess_single_image


def segment_characters(
    image_input: Union[str, Image.Image, np.ndarray],
    min_area: int = 30,
    space_ratio: float = 0.55,
) -> Tuple[List[dict], Image.Image]:
    """
    Segments handwritten text image into individual characters and identifies word spaces.
    
    Returns:
        segments: List of dicts with keys:
            'tensor': (1, 28, 28, 1) float32 normalized image for CNN,
            'bbox': (x_min, y_min, x_max, y_max),
            'space_after': bool (whether a space follows this character),
            'line_idx': int
        processed_source: Image used for segmentation.
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("L")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Ensure white text on black background
    np_img = np.array(pil_img)
    if np_img.mean() > 127:
        pil_img = ImageOps.invert(pil_img)
        np_img = np.array(pil_img)

    # Threshold binary image
    binary = np_img > 40

    # Slight dilation to connect broken pen strokes of the same character
    struct_elem = ndi.generate_binary_structure(2, 2)
    dilated = ndi.binary_dilation(binary, structure=struct_elem, iterations=1)

    # Label connected components
    labeled_array, num_features = ndi.label(dilated)
    slices = ndi.find_objects(labeled_array)

    raw_boxes = []
    for s in slices:
        y_slice, x_slice = s
        y_min, y_max = y_slice.start, y_slice.stop
        x_min, x_max = x_slice.start, x_slice.stop
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        if area >= min_area and width >= 3 and height >= 6:
            raw_boxes.append((x_min, y_min, x_max, y_max))

    if not raw_boxes:
        return [], pil_img

    # Sort boxes into text lines based on vertical overlap
    # Sort boxes from top to bottom first
    raw_boxes.sort(key=lambda b: b[1])

    lines = []
    for box in raw_boxes:
        y0, y1 = box[1], box[3]
        box_h = y1 - y0

        placed = False
        for line in lines:
            line_y0 = min(b[1] for b in line)
            line_y1 = max(b[3] for b in line)
            overlap = min(y1, line_y1) - max(y0, line_y0)
            # If vertical overlap is at least 30% of box height, they belong to the same line
            if overlap > box_h * 0.3:
                line.append(box)
                placed = True
                break

        if not placed:
            lines.append([box])

    # Sort lines from top to bottom
    lines.sort(key=lambda line: min(b[1] for b in line))

    # Within each line, sort horizontally from left to right
    for line in lines:
        line.sort(key=lambda b: b[0])

    # Calculate median character width for space detection
    all_widths = [b[2] - b[0] for b in raw_boxes]
    median_width = float(np.median(all_widths)) if all_widths else 20.0
    space_threshold = max(median_width * space_ratio, 10.0)

    segments = []
    for line_idx, line in enumerate(lines):
        for i, box in enumerate(line):
            x_min, y_min, x_max, y_max = box
            
            # Crop character from original binary image with a small padding
            pad = 2
            x0 = max(0, x_min - pad)
            y0 = max(0, y_min - pad)
            x1 = min(pil_img.width, x_max + pad)
            y1 = min(pil_img.height, y_max + pad)
            char_crop = pil_img.crop((x0, y0, x1, y1))

            # Preprocess crop to 28x28 normalized tensor
            tensor, _ = preprocess_single_image(char_crop, invert_colors=False)

            # Check for space after this character
            space_after = False
            if i < len(line) - 1:
                next_x_min = line[i + 1][0]
                gap = next_x_min - x_max
                if gap > space_threshold:
                    space_after = True
            elif line_idx < len(lines) - 1:
                # End of line gets a space
                space_after = True

            segments.append({
                "tensor": tensor,
                "bbox": box,
                "space_after": space_after,
                "line_idx": line_idx,
            })

    return segments, pil_img


def recognize_sentence(
    image_input: Union[str, Image.Image, np.ndarray],
    model: tf.keras.Model,
    label_map: Dict[int, str],
) -> Tuple[str, List[dict], Image.Image]:
    """
    End-to-end sentence recognition:
      1. Segments image into characters and spaces.
      2. Classifies each character using the CNN model.
      3. Reconstructs the complete recognized sentence string.
      4. Draws bounding boxes and labels on an annotated preview image.

    Returns:
        recognized_text: Reconstructed sentence string.
        char_details: List of dicts with character prediction, confidence, bbox.
        annotated_image: PIL Image with bounding boxes and predicted characters drawn.
    """
    segments, source_img = segment_characters(image_input)
    if not segments:
        return "", [], source_img

    # Prepare annotated image (RGB)
    annotated = source_img.convert("RGB")
    draw = ImageDraw.Draw(annotated)

    recognized_chars = []
    char_details = []

    for seg in segments:
        tensor = seg["tensor"]
        bbox = seg["bbox"]
        space_after = seg["space_after"]

        probs = model.predict(tensor, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class] * 100)
        char = label_map.get(pred_class, str(pred_class))

        recognized_chars.append(char)
        if space_after:
            recognized_chars.append(" ")

        char_details.append({
            "character": char,
            "confidence": confidence,
            "bbox": bbox,
            "space_after": space_after,
        })

        # Draw green bounding box and character annotation
        x0, y0, x1, y1 = bbox
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
        draw.text((x0, max(0, y0 - 15)), char, fill=(255, 255, 0))

    recognized_text = "".join(recognized_chars).strip()
    return recognized_text, char_details, annotated


def create_sample_sentence_image(
    text: str = "AI 2026",
    samples_dir: str = "dataset/samples",
    output_path: Optional[str] = "dataset/samples/sample_sentence.png",
) -> Image.Image:
    """
    Synthesizes a clean handwritten sentence/word image using available sample characters.
    Useful for testing sentence recognition out of the box.
    """
    char_images = []
    char_widths = []

    for ch in text:
        if ch == " ":
            # Space placeholder: 18x28 blank
            space_img = Image.new("L", (18, 28), 0)
            char_images.append(space_img)
            char_widths.append(18)
            continue

        sample_path = f"{samples_dir}/sample_{ch}.png"
        sample_path_upper = f"{samples_dir}/sample_{ch.upper()}.png"

        if os.path.exists(sample_path):
            img = Image.open(sample_path).convert("L")
        elif os.path.exists(sample_path_upper):
            img = Image.open(sample_path_upper).convert("L")
        else:
            # Fallback render using PIL default font
            img = Image.new("L", (28, 28), 0)
            d = ImageDraw.Draw(img)
            d.text((6, 2), ch, fill=255)

        char_images.append(img)
        char_widths.append(img.width)

    total_width = sum(char_widths) + len(char_images) * 4 + 20
    canvas_height = 48
    canvas = Image.new("L", (total_width, canvas_height), 0)

    cur_x = 10
    for img in char_images:
        y_offset = (canvas_height - img.height) // 2
        canvas.paste(img, (cur_x, y_offset))
        cur_x += img.width + 4

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        canvas.save(output_path)
        print(f"Sample sentence image saved to: {output_path}")

    return canvas
