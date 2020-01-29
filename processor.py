# -*- coding: utf-8 -*
import numpy
import numpy as np
from flyai.processor.base import Base
from flyai.processor.download import check_download
from skimage import io, transform

from path import DATA_PATH


class Processor(Base):

    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        # path 为图片的真实路径
        path = check_download(image_path, DATA_PATH)
        img = io.imread(path)
        img = np.array(img)  # 图片转化为矩阵向量
        input_x = transform.resize(img, output_shape=(224, 224))
        input_x = input_x.astype(np.float32)
        return input_x

    def input_y(self, label):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        input_y = [0]
        input_y.append(label)

        return input_y

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return numpy.argmax(data)