from nets.resnet50 import ResNet50,ResNet50_2
from tensorflow.keras.layers import BatchNormalization, Input, Flatten, Dense, Lambda, Activation,GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf




def euclidean_distance(vects):  # 欧氏距离
    """Find the Euclidean distance between two vectors.
    Arguments:
        vects: List containing two tensors of same length.
    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    dic = tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
    return dic


def cosine_distance(vects):
    """
    consine相似度：用两个向量的夹角判断两个向量的相似度，夹角越小，相似度越高，得到的consine相似度数值越大
    数值范围[-1,1],数值越大越相似。
    :param tensor1:
    :param tensor2:
    :return:
    """
    tensor1, tensor2 = vects
    # print(tensor1)

    # 求模长
    tensor1_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tensor1), axis=1, keepdims=True))
    tensor2_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tensor2), axis=1, keepdims=True))
    # print(tensor1_norm)

    # 内积
    tensor1_tensor2 = tf.math.reduce_sum(tf.math.multiply(tensor1, tensor2), axis=1, keepdims=True)
    # cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
    t1t2 = tf.math.multiply(tensor1_norm, tensor2_norm)
    cosin = tf.math.divide(tensor1_tensor2, t1t2)
    # print(cosin)

    return (1 - cosin)/2

def resnet_backbone(input, vgg_No):
    # vgg16主干网络
    if vgg_No == 1:
        feat1, feat2, feat3, feat4, feat5 = ResNet50(input)
    else:
        feat1, feat2, feat3, feat4, feat5 = ResNet50_2(input)
    x = Flatten()(feat5)
    # x = GlobalAveragePooling2D()(feat5)
    # mlp
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(256)(x)

    # x = BatchNormalization(name='mlpBN2')(x)
    # vgg输出
    return x

def self_sup_resnet(shape=( 576,1024, 3)):
    input_1 = Input(shape, name='self_input1')
    input_2 = Input(shape, name='self_input2')
    tower_1 = resnet_backbone(input_1, vgg_No=1)
    tower_2 = resnet_backbone(input_2, vgg_No=2)

    # merge_layer = Lambda(euclidean_distance)([tower_1, tower_2])
    merge_layer = Lambda(cosine_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization(name='self_output')(merge_layer)
    siamese = Model(inputs=[input_1, input_2], outputs=normal_layer)
    siamese.summary()
    return siamese

if __name__ == '__main__':
    self_sup_resnet(shape=(288,512,3))