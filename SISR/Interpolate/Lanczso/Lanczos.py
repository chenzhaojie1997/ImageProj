import cv2
import math
import numpy as np

def Lanczos_kernel(x:float, a:float=2) -> float:
    if abs(x) < 1e-6: 
        return 1
    elif abs(x) < a: 
        return a * math.sin(math.pi * x) * math.sin(math.pi * x / a) / (math.pi**2 * x**2)
    else:
        return 0

def interpolate_Lanczos_type(I:np.ndarray, scale:float) -> np.ndarray:
    assert len(I.shape) == 2
    h, w = I.shape
    nh, nw = int(h * scale), int(w * scale)
    Iu = np.zeros((nh, nw))

    for ny in range(nh):
        for nx in range(nw):
            ox, oy = nx / scale, ny / scale
            # 16 points for interpolate
            x0, y0 = int(ox) - 1, int(oy) - 1
            x1, y1 = int(ox),     int(oy)
            x2, y2 = int(ox) + 1, int(oy) + 1
            x3, y3 = int(ox) + 2, int(oy) + 2

            if (x0  < 0 or x3 >= w or y0 < 0 or y3 >= h): continue

            wx0 = Lanczos_kernel(ox - x0)
            wx1 = Lanczos_kernel(ox - x1)
            wx2 = Lanczos_kernel(ox - x2)
            wx3 = Lanczos_kernel(ox - x3)

            wy0 = Lanczos_kernel(oy - y0)
            wy1 = Lanczos_kernel(oy - y1)
            wy2 = Lanczos_kernel(oy - y2)
            wy3 = Lanczos_kernel(oy - y3)

            value = I[y0, x0] * wy0 * wx0 + \
                    I[y0, x1] * wy0 * wx1 + \
                    I[y0, x2] * wy0 * wx2 + \
                    I[y0, x3] * wy0 * wx3 + \
                    \
                    I[y1, x0] * wy1 * wx0 + \
                    I[y1, x1] * wy1 * wx1 + \
                    I[y1, x2] * wy1 * wx2 + \
                    I[y1, x3] * wy1 * wx3 + \
                    \
                    I[y2, x0] * wy2 * wx0 + \
                    I[y2, x1] * wy2 * wx1 + \
                    I[y2, x2] * wy2 * wx2 + \
                    I[y2, x3] * wy2 * wx3 + \
                    \
                    I[y3, x0] * wy3 * wx0 + \
                    I[y3, x1] * wy3 * wx1 + \
                    I[y3, x2] * wy3 * wx2 + \
                    I[y3, x3] * wy3 * wx3\

            Iu[ny, nx] = value

    return np.clip(Iu, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    I = cv2.imread('../data/set14/man.bmp', 0)
    # Iu = interpolate_Lanczos_type(I, 2)
    # cv2.imwrite('lanczos.bmp', Iu)

    Iu = cv2.resize(I, dsize=None, fx=2, fy=2, cv2.INTER_LANCZOS4)
    
