'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(image_path='images/cat.2718.jpg')
print(p)