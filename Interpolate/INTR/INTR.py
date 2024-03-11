import cv2
import numpy as np

# An Edge-Guided Image Interpolation Algorithm via Directional Filtering and Data Fusion

def INTR_X2(I:np.ndarray, m:int) -> np.ndarray:
    assert len(I.shape) == 2 and m % 2 == 1
    h, w = I.shape
    Iu = np.zeros((2 * h, 2 * w))

    # for pixel at [2x, 2y]
    for y in range(h):
        for x in range(w):
            Iu[2 * y, 2 * x] = I[y, x]

    # for pixel at [2x + 1, 2y + 1]
    for y in range(m // 2, h - m // 2):
        for x in range(m // 2, w - m // 2):
            pixels_45 = [] # pixels at 45°
            center = float(Iu[2 * y + 2, 2 * x] + Iu[2 * y, 2 * x + 2]) / 2 # center pixel
            pixels_45.append(center)

            is_inter = False
            for d in range(1, m // 2 + 1):
                dx, dy = d, -d
                if is_inter: # if lack, need interpolate
                    p1 = float(Iu[2 * y + 1 + dy - 1, 2 * x + 1 + dx + 1] + 
                               Iu[2 * y + 1 + dy + 1, 2 * x + 1 + dx - 1]) / 2
                    p2 = float(Iu[2 * y + 1 - dy - 1, 2 * x + 1 - dx + 1] + 
                               Iu[2 * y + 1 - dy + 1, 2 * x + 1 - dx - 1]) / 2
                    pixels_45.append(p1)
                    pixels_45.append(p2)
                else:
                    pixels_45.append(Iu[2 * y + 1 + dy, 2 * x + 1 + dx])
                    pixels_45.append(Iu[2 * y + 1 - dy, 2 * x + 1 - dx])
                is_inter = not is_inter
            
            pixels_135 = []
            center = float(Iu[2 * y, 2 * x] + Iu[2 * y + 2, 2 * x + 2]) / 2 # center pixel
            pixels_135.append(center)

            is_inter = False
            for d in range(1, m // 2 + 1):
                dx, dy = d, d
                if is_inter:
                    p1 = float(Iu[2 * y + 1 + dy - 1, 2 * x + 1 + dx - 1] + 
                               Iu[2 * y + 1 + dy + 1, 2 * x + 1 + dx + 1]) / 2
                    p2 = float(Iu[2 * y + 1 - dy - 1, 2 * x + 1 - dx - 1] + 
                               Iu[2 * y + 1 - dy + 1, 2 * x + 1 - dx - 1]) / 2
                    pixels_135.append(p1)
                    pixels_135.append(p2)
                else:
                    pixels_135.append(Iu[2 * y + 1 + dy, 2 * x + 1 + dx])
                    pixels_135.append(Iu[2 * y + 1 - dy, 2 * x + 1 - dx])
                is_inter = not is_inter
            
            # formula
            var_45 = np.var(pixels_45)
            var_135 = np.var(pixels_135)

            w_45 = var_135 / (var_45 + var_135 + 1e-6)
            w_135 = 1 - w_45

            Iu[2 * y + 1, 2 * x + 1] = w_45 * pixels_45[0] + w_135 * pixels_135[0]

    # for pixel at [2x, 2y + 1] or [2x + 1, 2y]
    for y in range(m // 2, h - m // 2):
        for x in range(m // 2, w - m // 2):
            for dx, dy in [(0, 1), (1, 0)]:
                px, py = 2 * x + dx, 2 * y + dy

                pixel_0 = [] # pixel at 0°
                center = float(Iu[py, px - 1] + Iu[py, px + 1]) / 2
                pixel_0.append(center)

                is_inter = False
                for d in range(1, m // 2 + 1):
                    if is_inter:
                        p1 = float(Iu[py, px - d - 1] + Iu[py, px - d + 1]) / 2
                        p2 = float(Iu[py, px + d - 1] + Iu[py, px + d + 1]) / 2
                        pixel_0.append(p1)
                        pixel_0.append(p2)
                    else:
                        pixel_0.append(Iu[py, px - d])
                        pixel_0.append(Iu[py, px + d])
                    is_inter = not is_inter
                
                pixel_180 = [] # pixel at 180°
                center = float(Iu[py - 1, px] + Iu[py + 1, px]) / 2
                pixel_180.append(center)

                is_inter = False
                for d in range(1, m // 2 + 1):
                    if is_inter:
                        p1 = float(Iu[py - d - 1, px] + Iu[py - d + 1, px]) / 2
                        p2 = float(Iu[py + d - 1, px] + Iu[py + d + 1, px]) / 2
                        pixel_180.append(p1)
                        pixel_180.append(p2)
                    else:
                        pixel_180.append(Iu[py - d, px])
                        pixel_180.append(Iu[py + d, px])
                    is_inter = not is_inter

                # formula
                var_0 = np.var(pixel_0)
                var_180 = np.var(pixel_180)

                w_0 = var_180 / (var_0 + var_180 + 1e-6)
                w_180 = 1 - w_0

                Iu[py, px] = w_0 * pixel_0[0] + w_180 * pixel_180[0]
    
    Iu = np.clip(Iu, 0, 255)
    # Ib = cv2.resize(I, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    # Iu[Iu == 0] = Ib[Iu == 0]
    return Iu

if __name__ == '__main__':
    I = cv2.imread('../data/set14/man.bmp', 0)
    Iu = INTR_X2(I, 9)
    cv2.imwrite('INTR_9.bmp', Iu)