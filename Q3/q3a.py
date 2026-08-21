import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Input image
# ============================================================

IMAGE_PATH = "data/hist/leh.png"


# ============================================================
# Linear Contrast Stretching
# ============================================================

def myLinearContrastStretch(img: np.ndarray):
    """
    Perform linear contrast stretching on the luminance
    component of a color image.

    g = (f - f_min) / (f_max - f_min)

    Input:
        img : BGR image

    Returns:
        stretched_img : BGR image
    """

    # --------------------------------------------------------
    # Convert BGR -> YCrCb
    # Y is the luminance component
    # --------------------------------------------------------
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    Y = ycrcb[:, :, 0].astype(np.float64)

    # --------------------------------------------------------
    # Find minimum and maximum luminance
    # --------------------------------------------------------
    f_min = np.min(Y)
    f_max = np.max(Y)

    # --------------------------------------------------------
    # Linear contrast stretching
    #
    # g = (f - f_min) / (f_max - f_min)
    #
    # Scale to [0, 255] because Y is an 8-bit image
    # --------------------------------------------------------
    if f_max == f_min:
        Y_stretched = Y.copy()
    else:
        Y_stretched = (
            (Y - f_min) / (f_max - f_min)
        ) * 255.0

    # Convert back to uint8
    Y_stretched = np.clip(Y_stretched, 0, 255).astype(np.uint8)

    # --------------------------------------------------------
    # Replace only luminance.
    # Keep original chroma components.
    # --------------------------------------------------------
    ycrcb[:, :, 0] = Y_stretched

    # --------------------------------------------------------
    # Convert YCrCb -> BGR
    # --------------------------------------------------------
    stretched_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return stretched_img


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------
    img = cv2.imread(IMAGE_PATH)

    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    # --------------------------------------------------------
    # Apply linear contrast stretching
    # --------------------------------------------------------
    stretched = myLinearContrastStretch(img)

    # --------------------------------------------------------
    # Convert BGR -> RGB for matplotlib
    # --------------------------------------------------------
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    stretched_rgb = cv2.cvtColor(stretched, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # Get luminance images for histograms
    # --------------------------------------------------------
    original_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    stretched_ycrcb = cv2.cvtColor(
        stretched,
        cv2.COLOR_BGR2YCrCb
    )

    original_Y = original_ycrcb[:, :, 0]
    stretched_Y = stretched_ycrcb[:, :, 0]

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")

    # --------------------------------------------------------
    # Contrast-stretched image
    # --------------------------------------------------------
    axes[0, 1].imshow(stretched_rgb)
    axes[0, 1].set_title("Linear Contrast Stretched Image")
    axes[0, 1].set_xlabel("Column")
    axes[0, 1].set_ylabel("Row")

    # --------------------------------------------------------
    # Original luminance histogram
    # --------------------------------------------------------
    axes[1, 0].hist(
        original_Y.ravel(),
        bins=256,
        range=(0, 256)
    )
    axes[1, 0].set_title("Histogram of Original Luminance")
    axes[1, 0].set_xlabel("Luminance")
    axes[1, 0].set_ylabel("Number of Pixels")
    axes[1, 0].set_xlim(0, 255)

    # --------------------------------------------------------
    # Stretched luminance histogram
    # --------------------------------------------------------
    axes[1, 1].hist(
        stretched_Y.ravel(),
        bins=256,
        range=(0, 256)
    )
    axes[1, 1].set_title(
        "Histogram of Contrast-Stretched Luminance"
    )
    axes[1, 1].set_xlabel("Luminance")
    axes[1, 1].set_ylabel("Number of Pixels")
    axes[1, 1].set_xlim(0, 255)

    plt.tight_layout()
    plt.show()