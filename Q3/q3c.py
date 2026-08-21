import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# INPUT IMAGES
# ============================================================

CANYON_PATH = "data/hist/canyon.png"
RETINA_PATH = "data/hist/retina.png"


# ============================================================
# TUNABLE CLAHE PARAMETERS
# ============================================================

# Main / tuned parameters
N_BINS = 64
WINDOW_SIZE = 101
HIST_THRESHOLD = 0.05

# Significantly larger neighborhood
LARGE_WINDOW_SIZE = 401

# Significantly smaller neighborhood
SMALL_WINDOW_SIZE = 21

# Half the histogram threshold
HALF_HIST_THRESHOLD = HIST_THRESHOLD / 2.0


# ============================================================
# CLAHE
# ============================================================

def myCLAHE(
    img: np.ndarray,
    n_bins: int,
    window_size: int,
    hist_threshold: float
):
    """
    Contrast-Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.

    n_bins : int
        Number of bins used for the local histogram.

    window_size : int
        Size of the square local neighborhood.

    hist_threshold : float
        Histogram clipping threshold in [0, 1].

        The threshold is interpreted as a fraction of the
        number of pixels in the local window.

    Returns
    -------
    enhanced_img : np.ndarray
        CLAHE enhanced BGR image.

    threshold_image : np.ndarray
        Per-pixel local clipping threshold.
    """

    # --------------------------------------------------------
    # Convert BGR -> YCrCb
    # --------------------------------------------------------

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    Y = ycrcb[:, :, 0].astype(np.float64)

    M, N = Y.shape

    # --------------------------------------------------------
    # Make sure window size is odd
    # --------------------------------------------------------

    if window_size % 2 == 0:
        window_size += 1

    half = window_size // 2

    # --------------------------------------------------------
    # Histogram clipping threshold
    #
    # Maximum number of pixels allowed in each histogram bin.
    #
    # hist_threshold = 0.05 means:
    #
    #     clip_limit = 5% of local window pixels
    #
    # --------------------------------------------------------

    window_area = window_size * window_size

    clip_limit = hist_threshold * window_area

    # Per-pixel threshold image
    threshold_image = np.full(
        (M, N),
        clip_limit,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Quantize luminance into histogram bins
    # --------------------------------------------------------

    bin_width = 256.0 / n_bins

    bin_indices = np.floor(Y / bin_width).astype(np.int32)

    # Make sure 255 belongs to the last bin
    bin_indices = np.clip(
        bin_indices,
        0,
        n_bins - 1
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    enhanced_Y = np.zeros(
        (M, N),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # We process the image one pixel at a time.
    #
    # For each pixel:
    #   1. Take its local window.
    #   2. Construct histogram.
    #   3. Clip histogram.
    #   4. Redistribute excess pixels.
    #   5. Compute CDF.
    #   6. Map the center pixel using the CDF.
    #
    # Boundary windows are automatically cropped.
    # --------------------------------------------------------

    for i in range(M):

        if i % 100 == 0:
            print(
                f"Processing row {i}/{M} "
                f"({100.0 * i / M:.1f}%)"
            )

        r0 = max(0, i - half)
        r1 = min(M, i + half + 1)

        for j in range(N):

            c0 = max(0, j - half)
            c1 = min(N, j + half + 1)

            # ------------------------------------------------
            # Local histogram
            # ------------------------------------------------

            local_bins = bin_indices[r0:r1, c0:c1]

            hist = np.bincount(
                local_bins.ravel(),
                minlength=n_bins
            ).astype(np.float64)

            # ------------------------------------------------
            # Actual number of pixels in the cropped window
            # ------------------------------------------------

            local_area = local_bins.size

            # ------------------------------------------------
            # Clip limit for this particular boundary window
            # ------------------------------------------------

            local_clip_limit = hist_threshold * local_area

            # Store per-pixel threshold
            threshold_image[i, j] = local_clip_limit

            # ------------------------------------------------
            # Clip histogram
            # ------------------------------------------------

            clipped = np.minimum(
                hist,
                local_clip_limit
            )

            # Number of excess pixels
            excess = np.sum(
                hist - clipped
            )

            # ------------------------------------------------
            # Redistribute excess uniformly
            # ------------------------------------------------

            clipped += excess / n_bins

            # ------------------------------------------------
            # Compute CDF
            # ------------------------------------------------

            center_bin = bin_indices[i, j]

            cdf = np.sum(
                clipped[:center_bin + 1]
            )

            # ------------------------------------------------
            # Normalize CDF
            # ------------------------------------------------

            # Use the actual local window size.
            #
            # This ensures that boundary windows are normalized
            # using only pixels actually present in the image.
            # ------------------------------------------------

            value = (
                cdf / local_area
            ) * 255.0

            enhanced_Y[i, j] = value

    # --------------------------------------------------------
    # Clip output luminance
    # --------------------------------------------------------

    enhanced_Y = np.clip(
        enhanced_Y,
        0,
        255
    )

    # --------------------------------------------------------
    # Replace luminance only
    #
    # Chroma is preserved.
    # --------------------------------------------------------

    ycrcb[:, :, 0] = enhanced_Y.astype(np.uint8)

    # --------------------------------------------------------
    # Convert back to BGR
    # --------------------------------------------------------

    enhanced_img = cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2BGR
    )

    return enhanced_img, threshold_image


# ============================================================
# LUMINANCE
# ============================================================

def get_luminance(img):

    ycrcb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2YCrCb
    )

    return ycrcb[:, :, 0]


# ============================================================
# DISPLAY ONE SET OF RESULTS
# ============================================================

def display_results(
    original,
    results,
    title_prefix
):

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Image display
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(results) + 1,
        figsize=(22, 6)
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    for k, (name, img) in enumerate(results, start=1):

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        axes[k].imshow(img_rgb)

        axes[k].set_title(name)

        axes[k].set_xlabel("Column")
        axes[k].set_ylabel("Row")

    fig.suptitle(
        title_prefix + " - CLAHE Results",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(results) + 1,
        figsize=(22, 5)
    )

    original_Y = get_luminance(original)

    axes[0].hist(
        original_Y.ravel(),
        bins=256,
        range=(0, 255)
    )

    axes[0].set_title("Original Histogram")
    axes[0].set_xlabel("Luminance")
    axes[0].set_ylabel("Number of Pixels")
    axes[0].set_xlim(0, 255)

    for k, (name, img) in enumerate(results, start=1):

        Y = get_luminance(img)

        axes[k].hist(
            Y.ravel(),
            bins=256,
            range=(0, 255)
        )

        axes[k].set_title(
            name + "\nHistogram"
        )

        axes[k].set_xlabel("Luminance")
        axes[k].set_ylabel("Number of Pixels")
        axes[k].set_xlim(0, 255)

    fig.suptitle(
        title_prefix + " - Luminance Histograms",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # CANYON
    # ========================================================

    canyon = cv2.imread(CANYON_PATH)

    if canyon is None:
        raise FileNotFoundError(
            f"Could not read {CANYON_PATH}"
        )

    print("\n========================================")
    print("Processing CANYON")
    print("========================================")

    # --------------------------------------------------------
    # 1. Tuned parameters
    # --------------------------------------------------------

    print("\nTuned CLAHE")

    canyon_normal, threshold_normal = myCLAHE(
        canyon,
        N_BINS,
        WINDOW_SIZE,
        HIST_THRESHOLD
    )

    # --------------------------------------------------------
    # 2. Large neighborhood
    # --------------------------------------------------------

    print("\nLarge neighborhood")

    canyon_large, threshold_large = myCLAHE(
        canyon,
        N_BINS,
        LARGE_WINDOW_SIZE,
        HIST_THRESHOLD
    )

    # --------------------------------------------------------
    # 3. Small neighborhood
    # --------------------------------------------------------

    print("\nSmall neighborhood")

    canyon_small, threshold_small = myCLAHE(
        canyon,
        N_BINS,
        SMALL_WINDOW_SIZE,
        HIST_THRESHOLD
    )

    # --------------------------------------------------------
    # 4. Half histogram threshold
    # --------------------------------------------------------

    print("\nHalf histogram threshold")

    canyon_half_threshold, threshold_half = myCLAHE(
        canyon,
        N_BINS,
        WINDOW_SIZE,
        HALF_HIST_THRESHOLD
    )

    # --------------------------------------------------------
    # Display Canyon
    # --------------------------------------------------------

    canyon_results = [

        (
            f"Tuned\n"
            f"Bins={N_BINS}, "
            f"W={WINDOW_SIZE}, "
            f"T={HIST_THRESHOLD}",
            canyon_normal
        ),

        (
            f"Large Window\n"
            f"W={LARGE_WINDOW_SIZE}",
            canyon_large
        ),

        (
            f"Small Window\n"
            f"W={SMALL_WINDOW_SIZE}",
            canyon_small
        ),

        (
            f"Half Threshold\n"
            f"T={HALF_HIST_THRESHOLD}",
            canyon_half_threshold
        )

    ]

    display_results(
        canyon,
        canyon_results,
        "Canyon"
    )

    # --------------------------------------------------------
    # Display per-pixel thresholds
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    threshold_data = [

        (
            "Tuned",
            threshold_normal
        ),

        (
            "Large Window",
            threshold_large
        ),

        (
            "Small Window",
            threshold_small
        ),

        (
            "Half Threshold",
            threshold_half
        )

    ]

    for ax, (name, threshold) in zip(
        axes,
        threshold_data
    ):

        im = ax.imshow(
            threshold,
            cmap="jet"
        )

        ax.set_title(
            "Local Histogram Threshold\n" + name
        )

        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        plt.colorbar(
            im,
            ax=ax
        )

    plt.tight_layout()
    plt.show()


    # ========================================================
    # RETINA
    # ========================================================

    retina = cv2.imread(RETINA_PATH)

    if retina is None:
        raise FileNotFoundError(
            f"Could not read {RETINA_PATH}"
        )

    print("\n========================================")
    print("Processing RETINA")
    print("========================================")

    # --------------------------------------------------------
    # Tuned CLAHE
    # --------------------------------------------------------

    retina_normal, retina_threshold = myCLAHE(
        retina,
        N_BINS,
        WINDOW_SIZE,
        HIST_THRESHOLD
    )

    # --------------------------------------------------------
    # Large neighborhood
    # --------------------------------------------------------

    retina_large, _ = myCLAHE(
        retina,
        N_BINS,
        LARGE_WINDOW_SIZE,
        HIST_THRESHOLD
    )

    # --------------------------------------------------------
    # Small neighborhood
    # --------------------------------------------------------

    retina_small, _ = myCLAHE(
        retina,
        N_BINS,
        SMALL_WINDOW_SIZE,
        HIST_THRESHOLD
    )

    # --------------------------------------------------------
    # Half histogram threshold
    # --------------------------------------------------------

    retina_half_threshold, _ = myCLAHE(
        retina,
        N_BINS,
        WINDOW_SIZE,
        HALF_HIST_THRESHOLD
    )

    # --------------------------------------------------------
    # Display Retina
    # --------------------------------------------------------

    retina_results = [

        (
            f"Tuned\n"
            f"Bins={N_BINS}, "
            f"W={WINDOW_SIZE}, "
            f"T={HIST_THRESHOLD}",
            retina_normal
        ),

        (
            f"Large Window\n"
            f"W={LARGE_WINDOW_SIZE}",
            retina_large
        ),

        (
            f"Small Window\n"
            f"W={SMALL_WINDOW_SIZE}",
            retina_small
        ),

        (
            f"Half Threshold\n"
            f"T={HALF_HIST_THRESHOLD}",
            retina_half_threshold
        )

    ]

    display_results(
        retina,
        retina_results,
        "Retina"
    )