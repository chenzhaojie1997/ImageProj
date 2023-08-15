import cv2
import numpy as np

# https://chiranjivi.tripod.com/EDITut.html
# https://github.com/Kirstihly/Edge-Directed_Interpolation

# up-sample 2 times by original NEDI
# I should be normalized to [0, 255] as type np.uint8
def NEDI_X2(I:np.ndarray, m:int) -> np.ndarray:
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
            # formula
            R = C.T @ C
            r = C.T @ Y
            a = np.linalg.pinv(R) @ r # pinv
            # 4 pixel arround center pixel; using diagonal pixel but small radius;
            # (0, 0) -> (2, 0), -> (2, 2) -> (0, 2)
            CC = np.zeros((1, 4))
            CC[0, 0] = Iu[2 * y,     2 * x]
            CC[0, 1] = Iu[2 * y + 2, 2 * x]
            CC[0, 2] = Iu[2 * y + 2, 2 * x + 2]
            CC[0, 3] = Iu[2 * y,     2 * x + 2]

            # get result
            Iu[2 * y + 1, 2 * x + 1] = CC @ a
    
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

                    # wrong
                    # Y[idx, 0] = Iu[2 * yi, 2 * xi]
                    # C[idx, 0] = Iu[2 * yi - 1, 2 * xi - 1]
                    # C[idx, 1] = Iu[2 * yi + 1, 2 * xi - 1]
                    # C[idx, 2] = Iu[2 * yi + 1, 2 * xi + 1]
                    # C[idx, 3] = Iu[2 * yi - 1, 2 * xi + 1]
                    idx += 1
            R = C.T @ C
            r = C.T @ Y
            a = np.linalg.pinv(R) @ r # pinv

            # for pixel at [2y + 1, 2x]
            CC = np.zeros((1, 4))
            CC[0, 0] = Iu[2 * y,     2 * x]
            CC[0, 1] = Iu[2 * y + 1, 2 * x - 1]
            CC[0, 2] = Iu[2 * y + 2, 2 * x]
            CC[0, 3] = Iu[2 * y + 1, 2 * x + 1]
            Iu[2 * y + 1, 2 * x] = CC @ a

            # for pixel at [2y, 2x + 1]
            CC[0, 0] = Iu[2 * y - 1, 2 * x + 1]
            CC[0, 1] = Iu[2 * y,     2 * x]
            CC[0, 2] = Iu[2 * y + 1, 2 * x + 1]
            CC[0, 3] = Iu[2 * y,     2 * x + 2]
            Iu[2 * y, 2 * x + 1] = CC @ a

    Iu = np.clip(Iu, 0, 255)
    Ib = cv2.resize(I, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    Iu[Iu == 0] = Ib[Iu == 0]
    return Iu

if __name__ == '__main__':
    from edi_git import EDI_predict

    I = cv2.imread('../data/set14/man.bmp', 0)
    # I = np.array(range(36)).reshape((6, 6))

    Iu = NEDI_X2(I, 4)
    cv2.imwrite('my.bmp', Iu)

    Iuu = EDI_predict(I, 4, 2)
    cv2.imwrite('test_git.bmp', Iuu)