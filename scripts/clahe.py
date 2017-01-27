"""
Implementation of contrast limited adaptive histogram equalization (CLAHE) to enhance an input image

Given an RGB image, convert the image to LAB color model and apply CLAHE on lightness channel.
Convert the CLAHE enhanced image back to RGB space and return the image
"""
import cv2
import numpy as np
from PIL import Image


def enhance(image_path, clip_limit=3):
    image = cv2.imread(image_path)
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert iamge from LAB color model back to RGB color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return cv2_to_pil(final_image)


def enhance_cv2(image_path, clip_limit=3):
    return pil_to_cv2(enhance(image_path, clip_limit=clip_limit))


def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_image):
    cv2_image = np.array(pil_image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    return cv2_image
