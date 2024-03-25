from __future__ import print_function
import numpy as np
import cv2
import os
import shutil
def rgb2ii(img, alpha):
    """Convert RGB image to illumination invariant image."""
    ii_image = (0.5 + np.log(img[:, :, 1] / float(255)) -
                alpha * np.log(img[:, :, 2] / float(255)) -
                (1 - alpha) * np.log(img[:, :, 0] / float(255)))

    return ii_image


if __name__ == "__main__":
    # simulated_sequences
    train_path = '/mnt/storage/home/lchen6/lchen6/data/Surgical/simulate_images/Scene1/'
    save_path = '/mnt/storage/home/lchen6/lchen6/data/Surgical/cii_images/'

    for root, dirs, imnames in os.walk(train_path):
        for imname in imnames:
            if ('im' in imname) & imname.endswith('.png'):
                source_img = cv2.imread(train_path + imname)
                a = 0.333  # Camera alpha
                invariant_img = rgb2ii(source_img, a)
                invariant_img /= np.amax(invariant_img)
                invariant_img=invariant_img*126
                saveimname = imname.replace('im', 'cii')
                cv2.imwrite(save_path+saveimname, invariant_img)
                # cv2.imshow("RGB Image", source_img)
                # cv2.imshow("Illumination Invariant", invariant_img)
                # cv2.waitKey()

    for root, dirs, imnames in os.walk(save_path):
        for imname in imnames:
            shutil.copyfile(os.path.join(save_path, imname), os.path.join(train_path, imname))

