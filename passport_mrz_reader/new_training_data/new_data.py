import random
import numpy as np
import os
import cv2
from PIL import Image

def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy



def list_files(dir):
    noises = ["gauss", "poisson"]
    i = 0
    for root, dirs, files in os.walk(dir):
        if len(files) == 0: continue
        if len(files) < 50:
            for i in range(50 - len(files)):
                image = cv2.imread(root + "/" + files[i % len(files)])
                root = root.replace("\\", "/")
                image = noisy(noises[random.randint(0, 1)], image)
                cv2.imwrite(f"{root}/GEN_{i}.jpeg", image)


list_files("data/images/train/final_trainingset")