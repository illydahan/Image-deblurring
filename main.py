import math

import cv2
import numpy as np

from PIL import Image
from scipy.ndimage.filters import convolve
from scipy.optimize import minimize
from matplotlib import pyplot as plt


def w_func(w, args=[]):
    # args={beta, v, alpha}
    return w**args[2] + 0.5*args[0] *((w-args[1]) **2)


def make_LUT(alpha):
    # 10k samples in range (-0.6,0.6)
    samples_count = 1e4
    look_up_table = 1e10 * np.ones(samples_count)
    step_value = 1.2 / samples_count
    beta = [math.pow(math.sqrt(2), i) for i in range(0,16)]

    index = 0
    for v in range(-0.6,0.6, step_value):
        min_val = 0
        for b in beta:
            min_val = minimize(w_func, np.array([0]), args=(b, v, alpha),
                               method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

            if look_up_table[index] < min_val : look_up_table[index] = min_val

        index += 1

    return look_up_table

def get_X_for_w(F, W, lamda, beta, K, y):
    f1, f2 = F[0], F[1]
    w1, w2 = W[0], W[1]

    numerator = np.dot(np.conj(np.fft.fft(f1)), np.fft.fft(w1)) + np.dot(np.conj(np.fft.fft(f2)), np.fft.fft(w2)) +  (lamda / beta) * np.dot(np.conj(np.fft.fft(K)), np.fft.fft(y))

    denominator = np.dot(np.conj(np.fft.fft(f1)), np.fft.fft(w1)) + np.dot(np.conj(np.fft.fft(f2)), np.fft.fft(f2)) +  (lamda / beta) * np.dot(np.conj(np.fft.fft(K)), np.fft.fft(K))

    return np.fft.ifft(numerator / denominator)




def get_w(x):
    return 1

def get_x(w):
    return 2


# Brief:  implementation of Algorithm 1 as stated by the article
# --------------------------------------------------------------
# Input:  Y                - blurred image
#         k                - blurring kernel
#         lamda            - regularization term
#         alpha            - exponent
#         b0, b_inc, b_max - regime parameters
#         T                - inner iterations count
# --------------------------------------------------------------
# Output: de-blurred image
# --------------------------------------------------------------
def deblur(Y,k, lamda, alpha, b0, b_inc, b_max, T):
    width, height = np.shape(Y)
    X = Y
    beta = b0
    alpha = 2/3

    LUT = make_LUT(alpha)

    while beta < b_max:
        iter = 0

        for i in range(T):
            w =

    #return X


def main():
    with Image.open("TestImages/lenna.tif") as img:
        blurred_image = blur(img)

        f, axarr = plt.subplots(3, 1)

        axarr[0].imshow(img, 'gray')
        axarr[1].imshow(blurred_image, 'gray')

        plt.show()
        print("done.")

def blur(clean_image):
    kernel = (1/9) * np.ones((3,3), dtype=np.float32)
    print(np.shape(kernel) , np.shape(clean_image))
    return convolve(clean_image, kernel, mode='nearest')

if __name__ == '__main__':
    main()

