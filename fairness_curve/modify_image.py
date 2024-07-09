import cv2
import numpy as np

def blur(filepath, amount=5, savepath=None):

    image = cv2.imread(filepath)
    blurred_image = cv2.GaussianBlur(image, (amount, amount), amount)
    cv2.imwrite(savepath, blurred_image)


def add_noise(filepath, mean=0, std=50, savepath=None):
    image = cv2.imread(filepath)
    noise = np.random.normal(mean, std, image.shape)
    noised_img = image + noise
    noised_img = np.clip(noised_img, 0, 255).astype(np.uint8)
    cv2.imwrite(savepath, noised_img)
    


blur("data/sptail_depth.png", 7, "data/blurred_spatial_depth.png")
# add_noise("data/sptail_depth.png", 0, 50, "data/noised_spatial_depth.png")

