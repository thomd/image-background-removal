import numpy as np
import cv2
from PIL import Image
import onnxruntime
import datetime
import os
import re
import base64
from io import BytesIO


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


def remove_background(**kwargs):
    if "path" in kwargs:
        img = cv2.imread(kwargs["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if "data" in kwargs:
        img = np.array(Image.open(BytesIO(kwargs["data"])))

    img_orig = img

    if len(img.shape) == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, 0:3]

    img = (img - 127.5) / 127.5  # normalize values to [-1, 1]
    img_h, img_w, img_c = img.shape
    x, y = get_scale_factor(img_h, img_w, 512)

    img = cv2.resize(img, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)

    img = np.transpose(img)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0).astype("float32")

    session = onnxruntime.InferenceSession("./model/pretrained/modnet.onnx", None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img})

    mask = (np.squeeze(result[0]) * 255).astype("uint8")
    mask = cv2.resize(mask, dsize=(img_w, img_h), interpolation=cv2.INTER_AREA)

    img_PIL = Image.fromarray(img_orig)
    mask = Image.fromarray(mask)
    img_PIL.putalpha(mask)

    buffered = BytesIO()
    img_PIL.save(buffered, format="PNG")
    img_PIL_base64 = base64.b64encode(buffered.getvalue())

    return img_PIL_base64.decode("ascii")
