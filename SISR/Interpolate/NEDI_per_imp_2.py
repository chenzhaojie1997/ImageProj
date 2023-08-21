import cv2
import numpy as np

# NEDI performance improvement 2
# iteratively refine super-resolution results with n iteration
def NEDI_per_imp_1(I:np.ndarray, m:int, n:int) -> np.ndarray:
    assert len(I.shape) == 2 and m % 2 == 0
    h, w = I.shape
    Iu = cv2.resize(I, dsize=None, fx=2, fy=2)

    for _ in range(n):
        pass

    return Iu