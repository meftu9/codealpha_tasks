"""
Data Preprocessing Module for Handwritten Character Recognition.
Supports both MNIST (digits 0-9) and EMNIST (alphanumeric digits 0-9 and letters A-Z).
Includes normalization, channel reshaping, label mapping, and image formatting.
"""

import json
import os
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


def load_raw_data(
    data_path: Optional[str] = None,
    dataset_type: str = "emnist",
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads raw handwritten dataset (EMNIST Balanced or MNIST).
    If dataset_type is 'emnist', loads the 47-class dataset (digits + uppercase/lowercase letters).
    If dataset_type is 'mnist', loads the 10-class digit dataset.
    """
    if data_path is None:
        if dataset_type.lower() == "emnist":
            data_path = "dataset/raw/emnist_balanced.npz"
        else:
            data_path = "dataset/raw/mnist.npz"

    if os.path.exists(data_path):
        print(f"Loading raw dataset from local file: {data_path}")
        with np.load(data_path, allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]
    else:
        if dataset_type.lower() == "emnist":
            from src.emnist_loader import download_and_extract_emnist_balanced
            print("EMNIST dataset not found locally. Downloading from NIST...")
            x_train, y_train, x_test, y_test, _ = download_and_extract_emnist_balanced(data_path)
        else:
            print("Loading dataset via tf.keras.datasets.mnist...")
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            print(f"Caching raw dataset to {data_path}...")
            np.savez_compressed(data_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    print(f"Loaded: x_train shape = {x_train.shape}, y_train shape = {y_train.shape}")
    print(f"Loaded: x_test shape = {x_test.shape}, y_test shape = {y_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalizes pixel values from [0, 255] to [0.0, 1.0]."""
    return images.astype("float32") / 255.0


def add_channel_dimension(images: np.ndarray) -> np.ndarray:
    """Ensures images have a trailing channel dimension (N, H, W, 1)."""
    if images.ndim == 3:
        return np.expand_dims(images, axis=-1)
    elif images.ndim == 4:
        return images
    else:
        raise ValueError(f"Expected 3D or 4D image array, got shape {images.shape}")


def preprocess_pipeline(
    x_train: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies normalization and adds channel dimension."""
    x_train_norm = normalize_images(x_train)
    x_test_norm = normalize_images(x_test)

    x_train_proc = add_channel_dimension(x_train_norm)
    x_test_proc = add_channel_dimension(x_test_norm)

    return x_train_proc, x_test_proc


def save_preprocessed_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str,
) -> None:
    """Saves preprocessed numpy arrays to a compressed .npz file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    print(f"Preprocessed features successfully saved to: {output_path}")


def load_preprocessed_data(
    npz_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads preprocessed numpy arrays from .npz file."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Features file not found at: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

    print(f"Loaded features from {npz_path}: x_train={x_train.shape}, x_test={x_test.shape}")
    return x_train, y_train, x_test, y_test


def save_label_mapping(mapping: Dict[str, str], filepath: str) -> None:
    """Saves class index to character mapping dictionary as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Label mapping saved to {filepath}")


def load_label_mapping(filepath: str) -> Dict[int, str]:
    """Loads class index to character mapping from JSON."""
    if not os.path.exists(filepath):
        return {i: str(i) for i in range(10)}

    with open(filepath, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    return {int(k): str(v) for k, v in raw_mapping.items()}


def create_default_mapping(num_classes: int = 47) -> Dict[str, str]:
    """
    Creates standard mapping for:
    - 10 classes: Digits 0-9
    - 47 classes: EMNIST Balanced (0-9 digits, A-Z uppercase, and merged distinctive lowercases)
    - 26 classes: Letters A-Z
    """
    if num_classes == 10:
        return {str(i): str(i) for i in range(10)}
    elif num_classes == 47:
        # Standard EMNIST Balanced mapping
        mapping = {str(i): str(i) for i in range(10)}
        for i in range(26):
            mapping[str(10 + i)] = chr(ord('A') + i)
        # 11 distinctive lowercase characters: a, b, d, e, f, g, h, n, q, r, t
        lowers = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
        for i, ch in enumerate(lowers):
            mapping[str(36 + i)] = ch
        return mapping
    elif num_classes == 26:
        return {str(i): chr(ord('A') + i) for i in range(26)}
    else:
        mapping = {str(i): str(i) for i in range(min(num_classes, 10))}
        for i in range(10, num_classes):
            mapping[str(i)] = chr(ord('A') + (i - 10))
        return mapping


def preprocess_single_image(
    image_input: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (28, 28),
    invert_colors: Optional[bool] = None,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Preprocesses a single image for CNN inference:
      1. Converts to Grayscale.
      2. Inverts colors if drawn on light background (white paper / black ink).
         EMNIST/MNIST models expect bright character on black background.
      3. Centers and resizes preserving aspect ratio to 20x20 inside 28x28 canvas.
      4. Normalizes to [0, 1] range.
      5. Reshapes to (1, 28, 28, 1).
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] in [3, 4]:
            pil_img = Image.fromarray(image_input).convert("L")
        else:
            pil_img = Image.fromarray(image_input.squeeze()).convert("L")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("L")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Check background inversion:
    np_raw = np.array(pil_img)
    if invert_colors is None:
        if np_raw.mean() > 127:
            pil_img = ImageOps.invert(pil_img)
    elif invert_colors:
        pil_img = ImageOps.invert(pil_img)

    # Crop bounding box of character to center it, then fit to 20x20 inside 28x28 canvas
    np_inv = np.array(pil_img)
    thresh = np_inv > 30
    if thresh.any():
        coords = np.argwhere(thresh)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = pil_img.crop((x0, y0, x1, y1))
        
        # Fit into 20x20 box preserving aspect ratio
        cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
        
        # Place centered in 28x28 black canvas
        final_img = Image.new("L", target_size, 0)
        paste_x = (target_size[0] - cropped.width) // 2
        paste_y = (target_size[1] - cropped.height) // 2
        final_img.paste(cropped, (paste_x, paste_y))
    else:
        final_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

    arr = np.array(final_img, dtype="float32") / 255.0
    tensor = arr.reshape(1, target_size[0], target_size[1], 1)

    return tensor, final_img
