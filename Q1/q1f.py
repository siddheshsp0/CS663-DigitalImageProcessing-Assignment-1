import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# Nearest Neighbor Interpolation
# ============================================================

def myNearestNeighborInterpolation(
    img: np.ndarray,
    M_new: int,
    N_new: int
):

    M, N = img.shape

    output = np.zeros(
        (M_new, N_new),
        dtype=img.dtype
    )

    for i in range(M_new):

        # Map output row to input coordinate
        x = i * (M - 1) / (M_new - 1)

        x_nearest = int(np.floor(x + 0.5))

        for j in range(N_new):

            # Map output column to input coordinate
            y = j * (N - 1) / (N_new - 1)

            y_nearest = int(np.floor(y + 0.5))

            output[i, j] = img[
                x_nearest,
                y_nearest
            ]

    return output


# ============================================================
# Bilinear Interpolation
# ============================================================

def myBilinearInterpolation(
    img: np.ndarray,
    M_new: int,
    N_new: int
):

    M, N = img.shape

    output = np.zeros(
        (M_new, N_new),
        dtype=np.float64
    )

    for i in range(M_new):

        # Corresponding coordinate in original image
        y = i * (M - 1) / (M_new - 1)

        y0 = int(np.floor(y))
        y1 = min(y0 + 1, M - 1)

        beta = y - y0

        for j in range(N_new):

            # Corresponding coordinate in original image
            x = j * (N - 1) / (N_new - 1)

            x0 = int(np.floor(x))
            x1 = min(x0 + 1, N - 1)

            alpha = x - x0

            # Four neighboring pixels
            I00 = img[y0, x0]
            I01 = img[y0, x1]
            I10 = img[y1, x0]
            I11 = img[y1, x1]

            # First interpolate along x
            top = (
                (1 - alpha) * I00
                + alpha * I01
            )

            bottom = (
                (1 - alpha) * I10
                + alpha * I11
            )

            # Then interpolate along y
            output[i, j] = (
                (1 - beta) * top
                + beta * bottom
            )

    return output


# ============================================================
# Bicubic Interpolation
#
# Follows the lecture-slide formulation:
#
# p(x,y) = sum_i=0^3 sum_j=0^3 a_ij x^i y^j
#
# At each corner:
# p    = f
# px   = fx
# py   = fy
# pxy  = fxy
#
# Derivatives are obtained using finite differences.
# ============================================================

def myBicubicInterpolation(
    img: np.ndarray,
    M_new: int,
    N_new: int
):

    M, N = img.shape

    output = np.zeros(
        (M_new, N_new),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Compute fx, fy and fxy at every pixel
    # --------------------------------------------------------

    fx = np.zeros(
        (M, N),
        dtype=np.float64
    )

    fy = np.zeros(
        (M, N),
        dtype=np.float64
    )

    fxy = np.zeros(
        (M, N),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # fx
    #
    # Interior:
    #
    # fx(x,y) = 0.5 [f(x+1,y) - f(x-1,y)]
    # --------------------------------------------------------

    fx[:, 1:-1] = 0.5 * (
        img[:, 2:] - img[:, :-2]
    )

    # Boundary: one-sided difference

    fx[:, 0] = (
        img[:, 1] - img[:, 0]
    )

    fx[:, -1] = (
        img[:, -1] - img[:, -2]
    )

    # --------------------------------------------------------
    # fy
    #
    # Interior:
    #
    # fy(x,y) = 0.5 [f(x,y+1) - f(x,y-1)]
    # --------------------------------------------------------

    fy[1:-1, :] = 0.5 * (
        img[2:, :] - img[:-2, :]
    )

    # Boundary: one-sided difference

    fy[0, :] = (
        img[1, :] - img[0, :]
    )

    fy[-1, :] = (
        img[-1, :] - img[-2, :]
    )

    # --------------------------------------------------------
    # fxy
    #
    # fxy(x,y) =
    #
    # 0.25 [
    #   f(x+1,y+1)
    #   + f(x-1,y-1)
    #   - f(x-1,y+1)
    #   - f(x+1,y-1)
    # ]
    # --------------------------------------------------------

    fxy[1:-1, 1:-1] = 0.25 * (
        img[2:, 2:]
        + img[:-2, :-2]
        - img[:-2, 2:]
        - img[2:, :-2]
    )

    # Boundary fxy using one-sided differences

    fxy[:, 0] = (
        fy[:, 1] - fy[:, 0]
    )

    fxy[:, -1] = (
        fy[:, -1] - fy[:, -2]
    )

    # --------------------------------------------------------
    # Construct 16 x 16 system
    #
    # Unknown ordering:
    #
    # a00 a01 a02 a03
    # a10 a11 a12 a13
    # a20 a21 a22 a23
    # a30 a31 a32 a33
    # --------------------------------------------------------

    A = np.zeros(
        (16, 16),
        dtype=np.float64
    )

    row = 0

    corners = [
        (0, 0),   # Q11
        (1, 0),   # Q21
        (0, 1),   # Q12
        (1, 1)    # Q22
    ]

    # --------------------------------------------------------
    # Basis for p(x,y)
    # --------------------------------------------------------

    def basis_p(x, y):

        return np.array([
            x**i * y**j
            for i in range(4)
            for j in range(4)
        ])

    # --------------------------------------------------------
    # Basis for px(x,y)
    # --------------------------------------------------------

    def basis_px(x, y):

        return np.array([
            i * x**(i - 1) * y**j
            if i >= 1 else 0.0
            for i in range(4)
            for j in range(4)
        ])

    # --------------------------------------------------------
    # Basis for py(x,y)
    # --------------------------------------------------------

    def basis_py(x, y):

        return np.array([
            j * x**i * y**(j - 1)
            if j >= 1 else 0.0
            for i in range(4)
            for j in range(4)
        ])

    # --------------------------------------------------------
    # Basis for pxy(x,y)
    # --------------------------------------------------------

    def basis_pxy(x, y):

        return np.array([
            i * j * x**(i - 1) * y**(j - 1)
            if i >= 1 and j >= 1
            else 0.0
            for i in range(4)
            for j in range(4)
        ])

    # --------------------------------------------------------
    # Build 16 equations
    # --------------------------------------------------------

    for x, y in corners:

        # p(x,y) = f(x,y)
        A[row, :] = basis_p(x, y)
        row += 1

        # px(x,y) = fx(x,y)
        A[row, :] = basis_px(x, y)
        row += 1

        # py(x,y) = fy(x,y)
        A[row, :] = basis_py(x, y)
        row += 1

        # pxy(x,y) = fxy(x,y)
        A[row, :] = basis_pxy(x, y)
        row += 1

    # Same A for every pixel cell
    A_inv = np.linalg.inv(A)

    # --------------------------------------------------------
    # Interpolate every output pixel
    # --------------------------------------------------------

    for i in range(M_new):

        # Coordinate in subsampled image
        y = (
            i * (M - 1)
            / (M_new - 1)
        )

        y1 = int(np.floor(y))
        y2 = min(y1 + 1, M - 1)

        # Local y coordinate in [0,1]
        v = y - y1

        for j in range(N_new):

            # Coordinate in subsampled image
            x = (
                j * (N - 1)
                / (N_new - 1)
            )

            x1 = int(np.floor(x))
            x2 = min(x1 + 1, N - 1)

            # Local x coordinate in [0,1]
            u = x - x1

            # ------------------------------------------------
            # Four corners
            # ------------------------------------------------

            b = np.array([

                # Q11
                img[y1, x1],
                fx[y1, x1],
                fy[y1, x1],
                fxy[y1, x1],

                # Q21
                img[y1, x2],
                fx[y1, x2],
                fy[y1, x2],
                fxy[y1, x2],

                # Q12
                img[y2, x1],
                fx[y2, x1],
                fy[y2, x1],
                fxy[y2, x1],

                # Q22
                img[y2, x2],
                fx[y2, x2],
                fy[y2, x2],
                fxy[y2, x2]

            ], dtype=np.float64)

            # ------------------------------------------------
            # Solve the 16 equations
            # ------------------------------------------------

            coefficients = A_inv @ b

            # ------------------------------------------------
            # Evaluate p(u,v)
            #
            # p(u,v) =
            # sum a_ij u^i v^j
            # ------------------------------------------------

            value = 0.0

            k = 0

            for ii in range(4):

                for jj in range(4):

                    value += (
                        coefficients[k]
                        * u**ii
                        * v**jj
                    )

                    k += 1

            output[i, j] = value

    return output


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load MATLAB .mat file
    # --------------------------------------------------------

    data = loadmat(
        "data/interp/ct.mat"
    )

    print("Variables in ct.mat:")
    print(data.keys())

    # --------------------------------------------------------
    # Extract images
    # --------------------------------------------------------

    original = np.asarray(
        data["original"],
        dtype=np.float64
    )

    subsampled = np.asarray(
        data["subsampled"],
        dtype=np.float64
    )

    print(
        "Original shape:",
        original.shape
    )

    print(
        "Subsampled shape:",
        subsampled.shape
    )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    M_original, N_original = original.shape

    M_subsampled, N_subsampled = subsampled.shape

    # --------------------------------------------------------
    # Enlarge subsampled image to EXACTLY the size
    # of the original image
    # --------------------------------------------------------

    enlarged_nn = myNearestNeighborInterpolation(
        subsampled,
        M_original,
        N_original
    )

    enlarged_bilinear = myBilinearInterpolation(
        subsampled,
        M_original,
        N_original
    )

    enlarged_bicubic = myBicubicInterpolation(
        subsampled,
        M_original,
        N_original
    )

    # --------------------------------------------------------
    # Difference images
    #
    # original - enlarged
    # --------------------------------------------------------

    difference_nn = (
        original - enlarged_nn
    )

    difference_bilinear = (
        original - enlarged_bilinear
    )

    difference_bicubic = (
        original - enlarged_bicubic
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse_nn = np.sqrt(
        np.mean(
            (original - enlarged_nn)**2
        )
    )

    rmse_bilinear = np.sqrt(
        np.mean(
            (original - enlarged_bilinear)**2
        )
    )

    rmse_bicubic = np.sqrt(
        np.mean(
            (original - enlarged_bicubic)**2
        )
    )

    print("\n================ RMSE ================\n")

    print(
        f"Nearest Neighbor : {rmse_nn:.6f}"
    )

    print(
        f"Bilinear         : {rmse_bilinear:.6f}"
    )

    print(
        f"Bicubic          : {rmse_bicubic:.6f}"
    )

    # --------------------------------------------------------
    # Common color limits for original + enlarged images
    # --------------------------------------------------------

    image_vmin = min(
        original.min(),
        enlarged_nn.min(),
        enlarged_bilinear.min(),
        enlarged_bicubic.min()
    )

    image_vmax = max(
        original.max(),
        enlarged_nn.max(),
        enlarged_bilinear.max(),
        enlarged_bicubic.max()
    )

    # --------------------------------------------------------
    # Common color limits for difference images
    #
    # Symmetric around zero makes positive/negative errors
    # comparable.
    # --------------------------------------------------------

    diff_max = max(
        np.abs(difference_nn).max(),
        np.abs(difference_bilinear).max(),
        np.abs(difference_bicubic).max()
    )

    diff_vmin = -diff_max
    diff_vmax = diff_max

    # ========================================================
    # Figure 1:
    # Original + three enlarged images
    # ========================================================

    fig1, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    im = axes[0].imshow(
        original,
        cmap="jet",
        vmin=image_vmin,
        vmax=image_vmax,
        aspect="equal"
    )

    axes[0].set_title(
        "Original"
    )

    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    axes[1].imshow(
        enlarged_nn,
        cmap="jet",
        vmin=image_vmin,
        vmax=image_vmax,
        aspect="equal"
    )

    axes[1].set_title(
        f"Nearest Neighbor\nRMSE = {rmse_nn:.4f}"
    )

    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    axes[2].imshow(
        enlarged_bilinear,
        cmap="jet",
        vmin=image_vmin,
        vmax=image_vmax,
        aspect="equal"
    )

    axes[2].set_title(
        f"Bilinear\nRMSE = {rmse_bilinear:.4f}"
    )

    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")

    axes[3].imshow(
        enlarged_bicubic,
        cmap="jet",
        vmin=image_vmin,
        vmax=image_vmax,
        aspect="equal"
    )

    axes[3].set_title(
        f"Bicubic\nRMSE = {rmse_bicubic:.4f}"
    )

    axes[3].set_xlabel("Column")
    axes[3].set_ylabel("Row")

    fig1.colorbar(
        im,
        ax=axes,
        shrink=0.8
    )

    fig1.suptitle(
        "CT Image: Original vs Interpolated Images",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

    # ========================================================
    # Figure 2:
    # Difference images
    # ========================================================

    fig2, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )

    im_diff = axes[0].imshow(
        difference_nn,
        cmap="jet",
        vmin=diff_vmin,
        vmax=diff_vmax,
        aspect="equal"
    )

    axes[0].set_title(
        "Original - Nearest Neighbor"
    )

    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    axes[1].imshow(
        difference_bilinear,
        cmap="jet",
        vmin=diff_vmin,
        vmax=diff_vmax,
        aspect="equal"
    )

    axes[1].set_title(
        "Original - Bilinear"
    )

    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    axes[2].imshow(
        difference_bicubic,
        cmap="jet",
        vmin=diff_vmin,
        vmax=diff_vmax,
        aspect="equal"
    )

    axes[2].set_title(
        "Original - Bicubic"
    )

    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")

    fig2.colorbar(
        im_diff,
        ax=axes,
        shrink=0.8
    )

    fig2.suptitle(
        "Interpolation Difference Images",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()