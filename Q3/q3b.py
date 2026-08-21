import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Input image
# ============================================================

IMAGE_PATH = "data/hist/leh.png"


# ============================================================
# Histogram Equalization
# ============================================================

def myHistEqualize(img: np.ndarray):
    """
    Perform histogram equalization on the luminance component.

    Input:
        img : RGB uint8 image

    Returns:
        equalized_rgb : RGB float32 image in [0, 1]
        Y_original    : original luminance
        Y_equalized   : equalized luminance
    """

    # --------------------------------------------------------
    # Convert RGB uint8 -> RGB float32 [0,1]
    # OpenCV cvtColor supports float32, NOT float64.
    # --------------------------------------------------------

    rgb = img.astype(np.float32) / 255.0

    # --------------------------------------------------------
    # RGB -> YCrCb
    # --------------------------------------------------------

    ycrcb = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2YCrCb
    )

    # --------------------------------------------------------
    # Extract luminance
    # --------------------------------------------------------

    Y = ycrcb[:, :, 0].astype(np.float64)

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    num_bins = 256

    histogram, bin_edges = np.histogram(
        Y.ravel(),
        bins=num_bins,
        range=(0.0, 1.0)
    )

    # --------------------------------------------------------
    # Cumulative Distribution Function
    # --------------------------------------------------------

    cdf = np.cumsum(histogram)

    # First non-zero histogram bin
    nonzero = np.nonzero(histogram)[0]

    if len(nonzero) == 0:
        return rgb, Y, Y

    cdf_min = cdf[nonzero[0]]

    # --------------------------------------------------------
    # Normalize CDF to [0,1]
    # --------------------------------------------------------

    cdf_normalized = (
        (cdf - cdf_min)
        / (Y.size - cdf_min)
    )

    cdf_normalized = np.clip(
        cdf_normalized,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Map every pixel through the CDF
    # --------------------------------------------------------

    bin_indices = np.floor(
        Y * num_bins
    ).astype(int)

    bin_indices = np.clip(
        bin_indices,
        0,
        num_bins - 1
    )

    Y_equalized = cdf_normalized[bin_indices]

    # --------------------------------------------------------
    # Put equalized luminance back
    # --------------------------------------------------------

    ycrcb_equalized = ycrcb.copy()

    # cv2 YCrCb float32 expects Y in [0,1]
    ycrcb_equalized[:, :, 0] = \
        Y_equalized.astype(np.float32)

    # --------------------------------------------------------
    # YCrCb -> RGB
    # --------------------------------------------------------

    equalized_rgb = cv2.cvtColor(
        ycrcb_equalized,
        cv2.COLOR_YCrCb2RGB
    )

    equalized_rgb = np.clip(
        equalized_rgb,
        0.0,
        1.0
    )

    return equalized_rgb, Y, Y_equalized

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    img_bgr = cv2.imread(IMAGE_PATH)

    if img_bgr is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    # OpenCV loads BGR -> convert to RGB
    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Histogram Equalization
    # --------------------------------------------------------

    equalized_rgb, original_Y, equalized_Y = \
        myHistEqualize(img_rgb)

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    # ========================================================
    # Original image
    # ========================================================

    axes[0, 0].imshow(img_rgb)

    axes[0, 0].set_title(
        "Original Image"
    )

    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")

    # ========================================================
    # Histogram-equalized image
    # ========================================================

    axes[0, 1].imshow(equalized_rgb)

    axes[0, 1].set_title(
        "Histogram Equalized Image"
    )

    axes[0, 1].set_xlabel("Column")
    axes[0, 1].set_ylabel("Row")

    # ========================================================
    # Original luminance histogram
    # ========================================================

    axes[1, 0].hist(
        original_Y.ravel(),
        bins=256,
        range=(0.0, 1.0)
    )

    axes[1, 0].set_title(
        "Histogram of Original Luminance"
    )

    axes[1, 0].set_xlabel("Luminance")
    axes[1, 0].set_ylabel("Number of Pixels")

    # ========================================================
    # Equalized luminance histogram
    # ========================================================

    axes[1, 1].hist(
        equalized_Y.ravel(),
        bins=256,
        range=(0.0, 1.0)
    )

    axes[1, 1].set_title(
        "Histogram of Equalized Luminance"
    )

    axes[1, 1].set_xlabel("Luminance")
    axes[1, 1].set_ylabel("Number of Pixels")

    plt.tight_layout()
    plt.show()