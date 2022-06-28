import colorsys
import copy
import time

import cv2
import numpy as np
from PIL import Image

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image

class Unet(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        #-------------------------------------------------------------------#
        "model_path"        : '/home/dell/python/Cherry_unsup/logs/h5_2022_04_23_19_39_24_对比分类15000_10epoch60601e41e5/ep120-loss0.707-val_loss0.635.h5',
        # "model_path": '/home/dell/python/Cherry_unsup/logs/h5_2022_05_09_12_20_16_迁移voc的restnet/ep087-loss0.327-val_loss0.492.h5',
        #----------------------------------------#
        #   所需要区分的类的个数+1
        #----------------------------------------#
        # "num_classes"       : 21,
        "num_classes": 5,
        #----------------------------------------#
        #   所使用的的主干网络：vgg、resnet50
        #----------------------------------------#
        "backbone"          : "resnet50",
        #----------------------------------------#
        #   输入图片的大小
        #----------------------------------------#
        # "input_shape"       : [512, 512],
        "input_shape" : [576, 1024],
        #----------------------------------------#
        #   blend参数用于控制是否
        #   让识别结果和原图混合
        #----------------------------------------#
        "blend"             : True,
    }

    #---------------------------------------------------#
    #   初始化UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
            # self.colors = [(0, 0, 0), (0, 0, 0), (255, 255, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0),
            #                (0, 0, 0)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   获得模型
        #---------------------------------------------------#
        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.model = unet([self.input_shape[0], self.input_shape[1], 3], self.num_classes, self.backbone)

        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, img_mask):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   归一化+添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        pr = self.model.predict(image_data)[0]
        #---------------------------------------------------#
        #   将灰条部分截取掉
        #---------------------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   进行图片的resize
        #---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        # 掩膜
        pr = cv2.bitwise_and(pr, pr, mask=img_mask)
        print(pr.shape)
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)
        print(pr.shape)


        #####计数
        #
        mask = [0,1,2,3,4]
        tmp = []
        for v in mask:
            tmp.append(np.sum(pr == v))

        tmp=np.array(tmp)
        tmp=tmp/(orininal_h*orininal_w)
        # print(tmp)
        # return tmp

        # ---------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        # ---------------------------------------------------#
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')

        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img))

        # ------------------------------------------------#
        #   将新图片和原图片混合
        # ------------------------------------------------#
        if self.blend:
            image = Image.blend(old_img, image, 0.7)

        image=cv2.cvtColor(np.asarray(image),cv2.COLOR_BGR2RGB)
        image = cv2.bitwise_and(image, image, mask=img_mask)
        image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

        return tmp,image