import cv2, os
import numpy as np

def blur(filepath, amount=5, savepath=None):

    image = cv2.imread(filepath)
    blurred_image = cv2.GaussianBlur(image, (amount, amount), 0)
    cv2.imwrite(savepath, blurred_image)


def add_noise(filepath, mean=0, std=50, savepath=None):
    image = cv2.imread(filepath)
    noise = np.random.normal(mean, std, image.shape)
    noised_img = image + noise
    noised_img = np.clip(noised_img, 0, 255).astype(np.uint8)
    cv2.imwrite(savepath, noised_img)
    

def modify_folder(folder, modify_type):
    files = os.listdir(folder)
    pre_transformed_files = [file for file in files if file.endswith('.png')]
    rest_of_path, last_component = os.path.split(folder)
    
    if modify_type == "noise":
        intensity = [15, 25, 40, 60, 70, 80]
        for i in intensity:
            new_folder_name = f"{modify_type}-{str(i)}-{last_component}"
            new_folder_full_path = os.path.join(rest_of_path, new_folder_name)
            if not os.path.exists(new_folder_full_path):
                os.mkdir(new_folder_full_path)
            for file in pre_transformed_files:
                add_noise(os.path.join(folder, file), 0, i, savepath=os.path.join(new_folder_full_path, file))

    if modify_type == "blur":
        intensity = [5, 11, 17, 23, 29, 35, 41]
        for i in intensity:
            
            new_folder_name = f"{modify_type}-{str(i)}-{last_component}"
            new_folder_full_path = os.path.join(rest_of_path, new_folder_name)
            if not os.path.exists(new_folder_full_path):
                os.mkdir(new_folder_full_path)
            for file in pre_transformed_files:
                blur(os.path.join(folder, file), i, savepath=os.path.join(new_folder_full_path, file))

modify_folder("data/poverty/no_noise", "blur")
modify_folder("data/poverty/no_noise", "noise")
# blur("data/sptail_depth.png", 61, "data/blurred_spatial_depth.png")
# add_noise("data/sptail_depth.png", 0, 140, "data/noised_spatial_depth.png")

