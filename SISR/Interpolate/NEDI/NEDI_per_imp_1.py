import cv2
import numpy as np

# NEDI performance improvement 1
# using covariance instread of dot
def NEDI_per_imp_1(I:np.ndarray, m:int) -> np.ndarray:
    assert len(I.shape) == 2 and m % 2 == 0
    h, w = I.shape
    Iu = np.zeros((2 * h, 2 * w))

    # for pixel at [2x, 2y]
    for y in range(h):
        for x in range(w):
            Iu[2 * y, 2 * x] = I[y, x]

    # for pixel at [2x + 1, 2y + 1]
    Y = np.zeros((m ** 2, 1))
    C = np.zeros((m ** 2, 4))
    for y in range(m//2, h - m//2):
        for x in range(m//2, w - m//2):
            idx = 0
            for yi in range(y - m//2, y + m//2):
                for xi in range(x - m//2, x + m//2):
                    # auxiliary pixel. It meaning is to express the pixel need to interpolate
                    # for pixel at [2x + 1, 2y + 1], using the window with the center [2x, 2y]
                    Y[idx, 0] = Iu[2 * yi, 2 * xi]
                    # using diagonal information to express every pixel in the window
                    # (-2, -2) -> (2, -2) -> (2, 2) -> (-2, 2)
                    C[idx, 0] = Iu[2 * yi - 2, 2 * xi - 2]
                    C[idx, 1] = Iu[2 * yi + 2, 2 * xi - 2]
                    C[idx, 2] = Iu[2 * yi + 2, 2 * xi + 2]
                    C[idx, 3] = Iu[2 * yi - 2, 2 * xi + 2]
                    idx += 1
            # covariance
            aver = np.sum(C, axis=0) / (m ** 2)
            C = C - aver
            aver = np.sum(Y) / (m ** 2)
            Y = Y - aver

            # formula
            R = C.T @ C
            r = C.T @ Y
            a = np.linalg.pinv(R) @ r # pinv

            CC = np.zeros((1, 4))
            CC[0, 0] = Iu[2 * y,     2 * x]
            CC[0, 1] = Iu[2 * y + 2, 2 * x]
            CC[0, 2] = Iu[2 * y + 2, 2 * x + 2]
            CC[0, 3] = Iu[2 * y,     2 * x + 2]

            # covar
            aver = np.sum(CC) / 4
            CC = CC - aver

            # get result
            Iu[2 * y + 1, 2 * x + 1] = CC @ a + aver
    
    # for pixel at [2x + 1, 2y] and [2x, 2y + 1]
    Y = np.zeros((m ** 2, 1))
    C = np.zeros((m ** 2, 4))
    for y in range(m//2, h - m//2):
        for x in range(m//2, w - m//2):
            idx  = 0
            for yi in range(y - m//2, y + m//2):
                for xi in range(x - m//2, x + m//2):
                    # auxiliary pixel. 
                    Y[idx, 0] = Iu[2 * yi + 1, 2 * xi - 1]
                    C[idx, 0] = Iu[2 * yi - 1, 2 * xi - 1]
                    C[idx, 1] = Iu[2 * yi + 1, 2 * xi - 3]
                    C[idx, 2] = Iu[2 * yi + 3, 2 * xi - 1]
                    C[idx, 3] = Iu[2 * yi + 1, 2 * xi + 1]
                    idx += 1
            # covariance
            aver = np.sum(C, axis=0) / (m ** 2)
            C = C - aver
            aver = np.sum(Y) / (m ** 2)
            Y = Y - aver

            R = C.T @ C
            r = C.T @ Y
            a = np.linalg.pinv(R) @ r # pinv

            # for pixel at [2y + 1, 2x]
            CC = np.zeros((1, 4))
            CC[0, 0] = Iu[2 * y,     2 * x]
            CC[0, 1] = Iu[2 * y + 1, 2 * x - 1]
            CC[0, 2] = Iu[2 * y + 2, 2 * x]
            CC[0, 3] = Iu[2 * y + 1, 2 * x + 1]

            # covar
            aver = np.sum(CC) / 4
            CC = CC - aver
            Iu[2 * y + 1, 2 * x] = CC @ a + aver

            # for pixel at [2y, 2x + 1]
            CC[0, 0] = Iu[2 * y - 1, 2 * x + 1]
            CC[0, 1] = Iu[2 * y,     2 * x]
            CC[0, 2] = Iu[2 * y + 1, 2 * x + 1]
            CC[0, 3] = Iu[2 * y,     2 * x + 2]

            # covar
            aver = np.sum(CC) / 4
            CC = CC - aver
            Iu[2 * y, 2 * x + 1] = CC @ a  + aver

    Iu = np.clip(Iu, 0, 255)
    Ib = cv2.resize(I, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    Iu[Iu == 0] = Ib[Iu == 0]
    return Iu

if __name__ == '__main__':
    from NEDI import NEDI_X2
    I = cv2.imread('../data/set14/man.bmp', 0)
    Iu = NEDI_per_imp_1(I, 4)
    cv2.imwrite('im1-8.bmp', Iu)
