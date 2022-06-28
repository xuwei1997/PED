from PIL import Image
import numpy as np
import os
import tensorflow as tf

def flip_random_crop(image):
    # With random crops we also apply horizontal flipping.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_crop(image, (288, 512, 3)) #!!
    return image


def color_jitter(x, strength=[0.7, 0.6, 0.6, 0.5]):
# def color_jitter(x, strength=[0.2, 0.2, 0.2, 0.08]):
    x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0, 255)
    return x


def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x


def custom_augment(image):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    image = flip_random_crop(image)
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image

def preprocess_input(image):
    image = image / 127.5 - 1
    return image


def one_img(img_path):
    input_shape = [432, 768]
    # 最后输出的还是 [288, 512]，有随机剪裁
    # input_shape = [576, 1024]
    # 有随机剪裁
    image = Image.open(img_path)
    h, w = input_shape
    image = image.resize((w, h), Image.BICUBIC)
    # print(image.size)
    image_np = np.array(image, np.float32)
    # print(image_np.shape)
    image_tf = custom_augment(image_np)
    # print(image_tf)

    # 恢复查看
    # img_tr = Image.fromarray(np.uint8(image_tf.numpy()))
    # img_tr.show()

    jpg = preprocess_input(image_tf)
    # print(jpg.shape)
    # print(jpg)

    return jpg


def double_img(dou_img):
    path = '/home/dell/out'
    # path = 'E:\charry\out'
    # path='/mnt/hdd/cherry2021/out'
    # print(dou_img)
    img0 = os.path.join(path, dou_img[0])
    img1 = os.path.join(path, dou_img[1])
    jpg0 = one_img(img0)
    jpg1 = one_img(img1)
    img_out = np.stack((jpg0, jpg1))
    # print(img_out.shape)
    return img_out


if __name__ == '__main__':
    # img_path='E:\charry\out\E88569964p1t09i0d2021-01-06_09.jpg'
    # j=one_img(img_path)
    img_path = 'E:\charry\out'
    double_img(['E88569964p1t15i3d2021-04-23_15.jpg', 'E88569964p1t12i0d2021-01-11_12.jpg'])
