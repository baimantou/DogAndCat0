import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH, LOG_PATH

# 导入flyai打印日志函数的库
from flyai.utils.log_helper import train_log
# 调用系统打印日志函数，这样在线上可看到训练和校验准确率和损失的实时变化曲线
 

# 数据获取辅助类
dataset = Dataset()

# 模型操作辅助类
model = Model(dataset)

'''
使用tensorflow实现自己的算法

'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

# 定义命名空间
x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_x')
y = tf.placeholder(tf.float32, [None, 2], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

x_image = tf.reshape(x, [-1, 224, 224, 3])


def vgg_19(inputs,
           num_classes=2,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False):
    with tf.variable_scope(scope, 'vgg_19', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            if num_classes:
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc8')
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            return net


g = tf.Graph()
prediction = tf.add(vgg_19(x_image), 0, name='y_conv')
loss = slim.losses.softmax_cross_entropy(prediction, y)  # 读入标签
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train_op = slim.learning.create_train_op(loss, optimizer)  # 训练以及优化

# 求准确率：
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.summary.merge_all()
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
    x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
    for i in range(args.EPOCHS):
        train_dict = {x: x_train, y: y_train, keep_prob: 0.7}
        sess.run(train_op, feed_dict=train_dict)
        losses, acc_ = sess.run([loss, accuracy], feed_dict=train_dict)
		train_log(train_loss=losses, train_acc=acc_)
        y_convs = sess.run(prediction, feed_dict=train_dict)
        print("step:{}, loss:{}, acc:{}".format(i + 1, losses, acc_))
        model.save_model(sess, MODEL_PATH, overwrite=True)