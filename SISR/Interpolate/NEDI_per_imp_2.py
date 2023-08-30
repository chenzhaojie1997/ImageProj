import cv2
import numpy as np

# NEDI performance improvement 2
# iteratively refine super-resolution results with n iteration
# not good
def NEDI_per_imp_2(I:np.ndarray, m:int, n:int) -> np.ndarray:
    assert len(I.shape) == 2 and m % 2 == 0
    h, w = I.shape

    Iu = cv2.resize(I, dsize=None, fx=2, fy=2)
    uh, uw = 2 * h, 2 * w

    Y = np.zeros((m ** 2, 1))
    C = np.zeros((m ** 2, 8))

    for _ in range(n):
        Iu_last = Iu.copy()
        for y in range(m // 2, uh - m // 2):
            for x in range(m // 2, uw - m // 2):
                idx = 0
                for iy in range(y - m // 2, y + m // 2):
                    for ix in range(x - m // 2, x + m // 2):
                        Y[idx, 0] = Iu_last[iy, ix]

                        C[idx, 0] = Iu_last[iy - 1, ix - 1]
                        C[idx, 1] = Iu_last[iy - 1, ix + 0]
                        C[idx, 2] = Iu_last[iy - 1, ix + 1]
                        C[idx, 3] = Iu_last[iy + 0, ix - 1]
                        C[idx, 4] = Iu_last[iy + 0, ix + 1]
                        C[idx, 5] = Iu_last[iy + 1, ix - 1]
                        C[idx, 6] = Iu_last[iy + 1, ix + 0]
                        C[idx, 7] = Iu_last[iy + 1, ix + 1]
                        idx += 1
                R = C.T @ C
                r = C.T @ Y
                a = np.linalg.pinv(R) @ r # pinv

                CC = np.zeros((1, 8))
                CC[0, 0] = Iu_last[iy - 1, ix - 1]
                CC[0, 1] = Iu_last[iy - 1, ix + 0]
                CC[0, 2] = Iu_last[iy - 1, ix + 1]
                CC[0, 3] = Iu_last[iy + 0, ix - 1]
                CC[0, 4] = Iu_last[iy + 0, ix + 1]
                CC[0, 5] = Iu_last[iy + 1, ix - 1]
                CC[0, 6] = Iu_last[iy + 1, ix + 0]
                CC[0, 7] = Iu_last[iy + 1, ix + 1]

                Iu[y, x] = CC @ a

    return Iu

if __name__ == '__main__':
    I = cv2.imread('../data/set14/man.bmp', 0)
    Iu = NEDI_per_imp_2(I, 4, 4)
    cv2.imwrite('./Imp2_4_4.bmp', Iu)
