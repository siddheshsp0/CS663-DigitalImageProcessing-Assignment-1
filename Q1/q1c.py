import numpy as np
import matplotlib.pyplot as plt
import cv2

IMG_PATH = "data/interp/random.png"

def myBilinearInterpolation(img: np.ndarray):
    M, N = img.shape

    # Required output dimensions
    M_new = 300 * (M - 1) + 1
    N_new = 300 * (N - 1) + 1

    # Use float so interpolation does not truncate values
    output = np.zeros((M_new, N_new), dtype=np.float64)

    for i in range(M_new):
        # Corresponding coordinate in original image
        y = i / 300.0

        y0 = int(np.floor(y))
        y1 = min(y0 + 1, M - 1)

        beta = y - y0

        for j in range(N_new):
            # Corresponding coordinate in original image
            x = j / 300.0

            x0 = int(np.floor(x))
            x1 = min(x0 + 1, N - 1)

            alpha = x - x0

            # Four neighboring pixels
            I00 = img[y0, x0]
            I01 = img[y0, x1]
            I10 = img[y1, x0]
            I11 = img[y1, x1]

            # Bilinear interpolation
            top = (1 - alpha) * I00 + alpha * I01
            bottom = (1 - alpha) * I10 + alpha * I11

            output[i, j] = (1 - beta) * top + beta * bottom

    return output






if __name__ == '__main__':
    # Read image using OpenCV
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

    # Check that image was loaded
    if img is None:
        raise FileNotFoundError("Could not read data/interp/random.png")

    # Bilinear interpolation
    resized = myBilinearInterpolation(img)


    # Display
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original
    im1 = axes[0].imshow(
        img,
        cmap="jet",
        aspect="equal",
        interpolation="nearest"
    )
    axes[0].set_title(
        f"Original Image ({img.shape[0]} × {img.shape[1]})"
    )
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    fig.colorbar(im1, ax=axes[0])


    # Resized
    im2 = axes[1].imshow(
        resized,
        cmap="jet",
        aspect="equal",
        interpolation="nearest"
    )
    axes[1].set_title(
        f"Bilinear Interpolation ({resized.shape[0]} × {resized.shape[1]})"
    )
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()