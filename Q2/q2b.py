import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# INPUT
# ============================================================

INPUT_IMAGE = "data/thresh/receipt.png"


# ============================================================
# Otsu Thresholding
# ============================================================

def myOtsuThreshold(img: np.ndarray):

    # Histogram
    histogram = np.bincount(
        img.ravel(),
        minlength=256
    ).astype(np.float64)

    # Normalize histogram
    histogram /= histogram.sum()

    # Intensity values
    intensity = np.arange(256)

    # Total image mean
    total_mean = np.sum(
        intensity * histogram
    )

    # Weight of class 0
    omega0 = 0.0

    # Cumulative mean of class 0
    mean0 = 0.0

    best_threshold = 0
    best_variance = -1.0

    # --------------------------------------------------------
    # Try every possible threshold
    # --------------------------------------------------------

    for T in range(256):

        omega0 += histogram[T]

        # Avoid empty class
        if omega0 == 0:
            continue

        omega1 = 1.0 - omega0

        # Avoid empty class
        if omega1 == 0:
            break

        mean0 += T * histogram[T]

        mu0 = mean0 / omega0

        mu1 = (
            total_mean - mean0
        ) / omega1

        # Between-class variance
        variance = (
            omega0
            * omega1
            * (mu0 - mu1) ** 2
        )

        if variance > best_variance:

            best_variance = variance
            best_threshold = T

    # --------------------------------------------------------
    # Binarize
    # --------------------------------------------------------

    binary = np.zeros_like(
        img,
        dtype=np.uint8
    )

    # Dark pixels -> black
    # Bright pixels -> white

    binary[
        img >= best_threshold
    ] = 255

    return binary, best_threshold


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

    # Otsu thresholding
    binary, threshold = myOtsuThreshold(
        img
    )

    print(
        f"Otsu threshold = {threshold}"
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
        f"Otsu Thresholding (T = {threshold})"
    )

    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    plt.tight_layout()
    plt.show()