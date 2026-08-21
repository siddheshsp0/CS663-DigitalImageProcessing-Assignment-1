import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# INPUT
# ============================================================

INPUT_IMAGE = "data/thresh/receipt.png"

# Tune this parameter
THRESHOLD = 150


# ============================================================
# Manual Thresholding
# ============================================================

def myManualThreshold(img: np.ndarray, threshold: float):

    binary = np.zeros_like(img, dtype=np.uint8)

    # Dark pixels -> black
    # Bright pixels -> white

    binary[img >= threshold] = 255

    return binary


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Read grayscale image
    img = cv2.imread(
        INPUT_IMAGE,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise FileNotFoundError(
            f"Could not read {INPUT_IMAGE}"
        )

    # Apply manual threshold
    binary = myManualThreshold(
        img,
        THRESHOLD
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    axes[0].imshow(
        img,
        cmap="gray",
        aspect="equal"
    )

    axes[0].set_title("Original")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    axes[1].imshow(
        binary,
        cmap="gray",
        vmin=0,
        vmax=255,
        aspect="equal"
    )

    axes[1].set_title(
        f"Manual Thresholding (T = {THRESHOLD})"
    )

    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    plt.tight_layout()
    plt.show()