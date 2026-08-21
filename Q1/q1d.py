import numpy as np
import cv2
import matplotlib.pyplot as plt


def myBicubicInterpolation(img: np.ndarray):

    M, N = img.shape

    # Required output dimensions
    M_new = 300 * (M - 1) + 1
    N_new = 300 * (N - 1) + 1

    # Bicubic interpolation produces real-valued values
    output = np.zeros((M_new, N_new), dtype=np.float64)

    # ------------------------------------------------------------
    # Compute fx, fy and fxy at every pixel using finite differences
    # ------------------------------------------------------------

    fx = np.zeros((M, N), dtype=np.float64)
    fy = np.zeros((M, N), dtype=np.float64)
    fxy = np.zeros((M, N), dtype=np.float64)

    # fx
    # Interior: central difference
    fx[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])

    # Boundary: one-sided difference
    fx[:, 0] = img[:, 1] - img[:, 0]
    fx[:, -1] = img[:, -1] - img[:, -2]

    # fy
    # Interior: central difference
    fy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])

    # Boundary: one-sided difference
    fy[0, :] = img[1, :] - img[0, :]
    fy[-1, :] = img[-1, :] - img[-2, :]

    # ------------------------------------------------------------
    # fxy
    #
    # Interior:
    #
    # fxy(x,y) =
    # 0.25 * [
    #       f(x+1,y+1) + f(x-1,y-1)
    #       - f(x-1,y+1) - f(x+1,y-1)
    # ]
    #
    # This is exactly the formula shown in the slides.
    # ------------------------------------------------------------

    fxy[1:-1, 1:-1] = 0.25 * (
        img[2:, 2:]
        + img[:-2, :-2]
        - img[:-2, 2:]
        - img[2:, :-2]
    )

    # For boundaries, compute fxy as the finite difference of fy
    # along x. This is the same finite-difference idea, with a
    # one-sided difference where a neighbor is unavailable.

    fxy[:, 0] = fy[:, 1] - fy[:, 0]
    fxy[:, -1] = fy[:, -1] - fy[:, -2]

    # ------------------------------------------------------------
    # Construct the 16 x 16 system for bicubic coefficients.
    #
    # p(x,y) = sum a_ij x^i y^j
    #
    # Unknown ordering:
    #
    # a00, a01, a02, a03,
    # a10, a11, a12, a13,
    # a20, a21, a22, a23,
    # a30, a31, a32, a33
    #
    # ------------------------------------------------------------

    A = np.zeros((16, 16), dtype=np.float64)

    row = 0

    # Four corners of unit square:
    #
    # Q11 = (0,0)
    # Q21 = (1,0)
    # Q12 = (0,1)
    # Q22 = (1,1)

    corners = [
        (0, 0),   # Q11
        (1, 0),   # Q21
        (0, 1),   # Q12
        (1, 1)    # Q22
    ]

    # ------------------------------------------------------------
    # Helper functions for the basis
    # ------------------------------------------------------------

    def basis_p(x, y):
        """
        Coefficients of a_ij in p(x,y)
        """
        return np.array([
            x**i * y**j
            for i in range(4)
            for j in range(4)
        ])

    def basis_px(x, y):
        """
        Coefficients of a_ij in p_x(x,y)
        """
        return np.array([
            (i * x**(i - 1) * y**j) if i >= 1 else 0.0
            for i in range(4)
            for j in range(4)
        ])

    def basis_py(x, y):
        """
        Coefficients of a_ij in p_y(x,y)
        """
        return np.array([
            (j * x**i * y**(j - 1)) if j >= 1 else 0.0
            for i in range(4)
            for j in range(4)
        ])

    def basis_pxy(x, y):
        """
        Coefficients of a_ij in p_xy(x,y)
        """
        return np.array([
            (i * j * x**(i - 1) * y**(j - 1))
            if i >= 1 and j >= 1 else 0.0
            for i in range(4)
            for j in range(4)
        ])

    # ------------------------------------------------------------
    # Build the 16 equations.
    #
    # At each corner:
    #
    # p       = f
    # px      = fx
    # py      = fy
    # pxy     = fxy
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # A is the same for every pixel cell.
    # Its inverse therefore only needs to be computed once.
    # ------------------------------------------------------------

    A_inv = np.linalg.inv(A)

    # ------------------------------------------------------------
    # Interpolate every output pixel
    # ------------------------------------------------------------

    for i in range(M_new):

        # Coordinate in original image
        y = i / 300.0

        y1 = int(np.floor(y))
        y2 = min(y1 + 1, M - 1)

        # Local coordinate inside the current unit square
        v = y - y1

        for j in range(N_new):

            # Coordinate in original image
            x = j / 300.0

            x1 = int(np.floor(x))
            x2 = min(x1 + 1, N - 1)

            # Local coordinate inside current unit square
            u = x - x1

            # ----------------------------------------------------
            # Data at the four corners
            #
            # Q11 = (x1,y1)
            # Q21 = (x2,y1)
            # Q12 = (x1,y2)
            # Q22 = (x2,y2)
            # ----------------------------------------------------

            b = np.array([
                img[y1, x1],
                fx[y1, x1],
                fy[y1, x1],
                fxy[y1, x1],

                img[y1, x2],
                fx[y1, x2],
                fy[y1, x2],
                fxy[y1, x2],

                img[y2, x1],
                fx[y2, x1],
                fy[y2, x1],
                fxy[y2, x1],

                img[y2, x2],
                fx[y2, x2],
                fy[y2, x2],
                fxy[y2, x2]
            ], dtype=np.float64)

            # ----------------------------------------------------
            # Solve the 16 equations for the 16 coefficients
            #
            # A @ coefficients = b
            # ----------------------------------------------------

            coefficients = A_inv @ b

            # ----------------------------------------------------
            # Evaluate p(u,v)
            #
            # p(u,v) = sum a_ij u^i v^j
            # ----------------------------------------------------

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




# Read image
img = cv2.imread(
    "data/interp/random.png",
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    raise FileNotFoundError(
        "Could not read data/interp/random.png"
    )

# Bicubic interpolation
resized = myBicubicInterpolation(img)

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


# Bicubic
im2 = axes[1].imshow(
    resized,
    cmap="jet",
    aspect="equal",
    interpolation="nearest"
)

axes[1].set_title(
    f"Bicubic Interpolation ({resized.shape[0]} × {resized.shape[1]})"
)
axes[1].set_xlabel("Column")
axes[1].set_ylabel("Row")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()