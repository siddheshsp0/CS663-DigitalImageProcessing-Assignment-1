# q1a.py
import cv2
import numpy as np
import matplotlib.pyplot as plt


IMG_PATH = "./data/interp/random.png"


def myBicubicInterpolation(img: np.ndarray):
    M, N = img.shape
    M_new = 300 * (M - 1) + 1
    N_new = 300 * (N - 1) + 1
    output = np.zeros((M_new, N_new), dtype=np.float64)
    # 299 pixels between two original pixels
    fx = np.zeros((M, N), dtype=np.float64)
    fy = np.zeros((M, N), dtype=np.float64)
    fxy = np.zeros((M, N), dtype=np.float64)

    img = img.astype(np.float64)

    # fx
    fx[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
    fx[0, :] = img[1, :] - img[0, :]
    fx[-1, :] = img[-1, :] - img[-2, :]

    # fy
    fy[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    fy[:, 0] = img[:, 1] - img[:, 0]
    fy[:, -1] = img[:, -1] - img[:, -2]

    # fxy
    fxy[:, 1:-1] = 0.5 * (fx[:, 2:] - fx[:, :-2])
    fxy[:, 0] = fx[:, 1] - fx[:, 0]
    fxy[:, -1] = fx[:, -1] - fx[:, -2]


    # Calculated matrix to solve the 16 equations
    Mmat = np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],
        [-3, 3, -2, -1],
        [2, -2,  1,  1]
    ], dtype=np.float64)

    for i in range(M - 1):
        for j in range(N - 1):

            # Four corners of current square
            x0, x1 = i, i + 1
            y0, y1 = j, j + 1

            # -------------------------------------------------
            # G contains function values and derivatives
            #
            #       y=0          y=1
            #
            # x=0   f00, fy00    f01, fy01
            #       fx00,fxy00   fx01,fxy01
            #
            # x=1   f10, fy10    f11, fy11
            #       fx10,fxy10   fx11,fxy11
            # -------------------------------------------------

            G = np.array([
                [img[x0, y0],   fy[x0, y0],
                 img[x0, y1],   fy[x0, y1]],

                [fx[x0, y0],    fxy[x0, y0],
                 fx[x0, y1],    fxy[x0, y1]],

                [img[x1, y0],   fy[x1, y0],
                 img[x1, y1],   fy[x1, y1]],

                [fx[x1, y0],    fxy[x1, y0],
                 fx[x1, y1],    fxy[x1, y1]]
            ])

            # Bicubic polynomial coefficient matrix
            A = Mmat @ G @ Mmat.T

            for u in range(301):
                if u == 300 and i != M - 2:
                    continue
                x = u / 300.0
                X = np.array([1, x, x**2, x**3])

                for v in range(301):
                    if v == 300 and j != N - 2:
                        continue
                    y = v / 300.0

                    Y = np.array([1, y, y**2, y**3])
                    # p(x,y) = X A Y^T
                    value = X @ A @ Y

                    output[300 * i + u, 300 * j + v] = value

    return output



def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

    img_ = myBicubicInterpolation(img)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original
    im1 = axes[0].imshow(img, cmap="jet")
    axes[0].set_title("Original")
    axes[0].set_xlabel("x (pixels)")
    axes[0].set_ylabel("y (pixels)")
    axes[0].set_aspect("equal")
    fig.colorbar(im1, ax=axes[0])
    # Interpolated
    im2 = axes[1].imshow(img_, cmap="jet")
    axes[1].set_title("Bicubic Interpolation")
    axes[1].set_xlabel("x (pixels)")
    axes[1].set_ylabel("y (pixels)")
    axes[1].set_aspect("equal")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()
