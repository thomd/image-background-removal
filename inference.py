import numpy as np
import cv2
from PIL import Image
import onnxruntime
import datetime
import os
import re

def get_scale_factor(im_h, im_w, ref_size):
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h
    return x_scale_factor, y_scale_factor

def remove_background(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    im = (im - 127.5) / 127.5 # normalize values to [-1, 1]
    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, 512)

    im = cv2.resize(im, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)

    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis=0).astype("float32")

    session = onnxruntime.InferenceSession("./model/pretrained/modnet.onnx", None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    mask = (np.squeeze(result[0]) * 255).astype("uint8")
    mask = cv2.resize(mask, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)

    # cv2.imwrite(f"output_{datetime.datetime.now()}.png", mask)

    im_PIL = Image.open(image_path)
    mask = Image.fromarray(mask)
    im_PIL.putalpha(mask)
    p = re.compile('(.*)\.(jpg|jpeg|png|git)$')
    cropped_image_path = p.sub(r'\1_cropped.png', image_path)
    im_PIL.save(cropped_image_path)
