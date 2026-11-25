"""
EMNIST Loader using Remote HTTP Range Requests to download only the necessary balanced split
(approx 31 MB total instead of 561 MB).
"""

import gzip
import io
import os
import struct
import urllib.request
import zipfile
import numpy as np


class RemoteZipFile(io.RawIOBase):
    """File-like object that fetches byte ranges from an HTTP server on demand."""
    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            self.size = int(resp.headers.get("Content-Length"))
        self.pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = self.size - self.pos
        if self.pos >= self.size or size <= 0:
            return b""
        end = min(self.pos + size - 1, self.size - 1)
        req = urllib.request.Request(
            self.url,
            headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={self.pos}-{end}"}
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def parse_idx_images(gz_bytes: bytes) -> np.ndarray:
    """Parses idx3-ubyte gzipped image bytes and transposes to standard orientation."""
    with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes)) as gz:
        magic, num_images, rows, cols = struct.unpack(">IIII", gz.read(16))
        buffer = gz.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        # EMNIST is column-major: transpose each 28x28 image to row-major
        data = np.transpose(data, (0, 2, 1))
    return data


def parse_idx_labels(gz_bytes: bytes) -> np.ndarray:
    """Parses idx1-ubyte gzipped label bytes."""
    with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes)) as gz:
        magic, num_labels = struct.unpack(">II", gz.read(8))
        buffer = gz.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    return labels


def parse_mapping(mapping_str: str) -> dict:
    """Parses EMNIST mapping text file into integer class -> character."""
    mapping = {}
    for line in mapping_str.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            cls_idx = int(parts[0])
            ascii_code = int(parts[1])
            mapping[cls_idx] = chr(ascii_code)
    return mapping


def download_and_extract_emnist_balanced(output_npz: str = "dataset/raw/emnist_balanced.npz"):
    """
    Downloads EMNIST Balanced (47 classes: 0-9 digits + A-Z letters) via remote zip
    and saves to compressed NPZ.
    """
    if os.path.exists(output_npz):
        print(f"EMNIST Balanced dataset already exists at {output_npz}")
        with np.load(output_npz, allow_pickle=True) as data:
            return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    print(f"Connecting to EMNIST archive at {url}...")
    remote = RemoteZipFile(url)

    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    with zipfile.ZipFile(remote) as zf:
        print("Downloading EMNIST Balanced mapping...")
        mapping_str = zf.read("gzip/emnist-balanced-mapping.txt").decode("utf-8")
        mapping = parse_mapping(mapping_str)

        print("Downloading EMNIST Balanced training images (~26.9 MB)...")
        train_img_gz = zf.read("gzip/emnist-balanced-train-images-idx3-ubyte.gz")
        print("Parsing training images...")
        x_train = parse_idx_images(train_img_gz)

        print("Downloading EMNIST Balanced training labels (~80 KB)...")
        train_lbl_gz = zf.read("gzip/emnist-balanced-train-labels-idx1-ubyte.gz")
        y_train = parse_idx_labels(train_lbl_gz)

        print("Downloading EMNIST Balanced testing images (~4.5 MB)...")
        test_img_gz = zf.read("gzip/emnist-balanced-test-images-idx3-ubyte.gz")
        print("Parsing testing images...")
        x_test = parse_idx_images(test_img_gz)

        print("Downloading EMNIST Balanced testing labels (~10 KB)...")
        test_lbl_gz = zf.read("gzip/emnist-balanced-test-labels-idx1-ubyte.gz")
        y_test = parse_idx_labels(test_lbl_gz)

    print(f"Saving extracted EMNIST to {output_npz}...")
    np.savez_compressed(
        output_npz,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    print(f"Successfully saved {output_npz} (Train: {x_train.shape}, Test: {x_test.shape})")
    return x_train, y_train, x_test, y_test, mapping


if __name__ == "__main__":
    download_and_extract_emnist_balanced()
