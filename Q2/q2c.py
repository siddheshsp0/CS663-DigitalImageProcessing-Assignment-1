import cv2
import numpy as np
import matplotlib.pyplot as plt


INPUT_IMAGE = "data/thresh/receipt.png"

# Must be odd
BLOCK_SIZE = 31

# Threshold offset
C = 5

# Local / Adaptive Thresholding
def myAdaptiveThreshold(img, block_size, C):

    if block_size % 2 == 0:
        raise ValueError(
            "block_size must be odd"
        )

    if block_size < 3:
        raise ValueError(
            "block_size must be >= 3"
        )

    # --------------------------------------------------------
    # Compute local mean
    #
    # cv2.boxFilter computes the average over a local
    # block around every pixel.
    # --------------------------------------------------------

    local_mean = cv2.boxFilter(
        img.astype(np.float64),
        ddepth=-1,
        ksize=(block_size, block_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT
    )

    # Per-pixel threshold
    # T(x,y) = local_mean(x,y) - C

    threshold_image = (local_mean - C)

    # Apply threshold
    # Darker than local threshold -> black
    # Brighter than local threshold -> white

    binary = np.zeros_like(img,dtype=np.uint8)

    binary[img >= threshold_image] = 255

    return binary, threshold_image

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

    # Adaptive thresholding
    binary, threshold_image = myAdaptiveThreshold(
        img,
        BLOCK_SIZE,
        C
    )

    # --------------------------------------------------------
    # Display
    #
    # Original
    # Thresholded
    # Per-pixel threshold values
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5)
    )

    # Original
    axes[0].imshow(
        img,
        cmap="gray",
        aspect="equal"
    )

    axes[0].set_title(
        "Original"
    )

    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    # Thresholded
    axes[1].imshow(
        binary,
        cmap="gray",
        vmin=0,
        vmax=255,
        aspect="equal"
    )

    axes[1].set_title(
        f"Adaptive Thresholding\n"
        f"Block = {BLOCK_SIZE}, C = {C}"
    )

    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    # Per-pixel threshold
    im = axes[2].imshow(
        threshold_image,
        cmap="jet",
        aspect="equal"
    )

    axes[2].set_title(
        "Per-Pixel Threshold Values"
    )

    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")

    fig.colorbar(
        im,
        ax=axes[2],
        label="Threshold"
    )

    plt.tight_layout()
    plt.show()