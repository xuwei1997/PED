from nets.resnet50 import ResNet50,ResNet50_2
from tensorflow.keras.layers import BatchNormalization, Input, Flatten, Dense, Lambda, Activation,GlobalAveragePooling2D,subtract
from tensorflow.keras.models import Model
import tensorflow as tf

def distance(vects):  # 欧氏距离
    """Find the Euclidean distance between two vectors.
    Arguments:
        vects: List containing two tensors of same length.
    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    # print(vects)
    x = subtract(vects)
    # print(x)
    x = tf.math.square(x)
    # print(x)
    return x






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
    return x

def self_sup_resnet(shape=( 576,1024, 3)):
    input_1 = Input(shape, name='self_input1')
    input_2 = Input(shape, name='self_input2')
    tower_1 = resnet_backbone(input_1, vgg_No=1)
    tower_2 = resnet_backbone(input_2, vgg_No=2)

    x = Lambda(distance)([tower_1, tower_2])
    # x=Dense(128)(merge_layer)
    x=Dense(4,activation='softmax',name='self_output')(x)

    siamese = Model(inputs=[input_1, input_2], outputs=x)

    siamese.get_layer(name='conv1_2').set_weights(siamese.get_layer(name='conv1').get_weights())
    siamese.get_layer(name='bn_conv1_2').set_weights(siamese.get_layer(name='bn_conv1').get_weights())

    #2a
    siamese.get_layer(name='res20a_branch2a').set_weights(siamese.get_layer(name='res2a_branch2a').get_weights())
    siamese.get_layer(name='bn20a_branch2a').set_weights(siamese.get_layer(name='bn2a_branch2a').get_weights())
    siamese.get_layer(name='res20a_branch2b').set_weights(siamese.get_layer(name='res2a_branch2b').get_weights())
    siamese.get_layer(name='bn20a_branch2b').set_weights(siamese.get_layer(name='bn2a_branch2b').get_weights())
    siamese.get_layer(name='res20a_branch2c').set_weights(siamese.get_layer(name='res2a_branch2c').get_weights())
    siamese.get_layer(name='bn20a_branch2c').set_weights(siamese.get_layer(name='bn2a_branch2c').get_weights())
    siamese.get_layer(name='res20a_branch1').set_weights(siamese.get_layer(name='res2a_branch1').get_weights())
    siamese.get_layer(name='bn20a_branch1').set_weights(siamese.get_layer(name='bn2a_branch1').get_weights())

    #2b
    siamese.get_layer(name='res20b_branch2a').set_weights(siamese.get_layer(name='res2b_branch2a').get_weights())
    siamese.get_layer(name='bn20b_branch2a').set_weights(siamese.get_layer(name='bn2b_branch2a').get_weights())
    siamese.get_layer(name='res20b_branch2b').set_weights(siamese.get_layer(name='res2b_branch2b').get_weights())
    siamese.get_layer(name='bn20b_branch2b').set_weights(siamese.get_layer(name='bn2b_branch2b').get_weights())
    siamese.get_layer(name='res20b_branch2c').set_weights(siamese.get_layer(name='res2b_branch2c').get_weights())
    siamese.get_layer(name='bn20b_branch2c').set_weights(siamese.get_layer(name='bn2b_branch2c').get_weights())

    # 2c
    siamese.get_layer(name='res20c_branch2a').set_weights(siamese.get_layer(name='res2c_branch2a').get_weights())
    siamese.get_layer(name='bn20c_branch2a').set_weights(siamese.get_layer(name='bn2c_branch2a').get_weights())
    siamese.get_layer(name='res20c_branch2b').set_weights(siamese.get_layer(name='res2c_branch2b').get_weights())
    siamese.get_layer(name='bn20c_branch2b').set_weights(siamese.get_layer(name='bn2c_branch2b').get_weights())
    siamese.get_layer(name='res20c_branch2c').set_weights(siamese.get_layer(name='res2c_branch2c').get_weights())
    siamese.get_layer(name='bn20c_branch2c').set_weights(siamese.get_layer(name='bn2c_branch2c').get_weights())

    # 3a
    siamese.get_layer(name='res30a_branch2a').set_weights(siamese.get_layer(name='res3a_branch2a').get_weights())
    siamese.get_layer(name='bn30a_branch2a').set_weights(siamese.get_layer(name='bn3a_branch2a').get_weights())
    siamese.get_layer(name='res30a_branch2b').set_weights(siamese.get_layer(name='res3a_branch2b').get_weights())
    siamese.get_layer(name='bn30a_branch2b').set_weights(siamese.get_layer(name='bn3a_branch2b').get_weights())
    siamese.get_layer(name='res30a_branch2c').set_weights(siamese.get_layer(name='res3a_branch2c').get_weights())
    siamese.get_layer(name='bn30a_branch2c').set_weights(siamese.get_layer(name='bn3a_branch2c').get_weights())
    siamese.get_layer(name='res30a_branch1').set_weights(siamese.get_layer(name='res3a_branch1').get_weights())
    siamese.get_layer(name='bn30a_branch1').set_weights(siamese.get_layer(name='bn3a_branch1').get_weights())

    # 3b
    siamese.get_layer(name='res30b_branch2a').set_weights(siamese.get_layer(name='res3b_branch2a').get_weights())
    siamese.get_layer(name='bn30b_branch2a').set_weights(siamese.get_layer(name='bn3b_branch2a').get_weights())
    siamese.get_layer(name='res30b_branch2b').set_weights(siamese.get_layer(name='res3b_branch2b').get_weights())
    siamese.get_layer(name='bn30b_branch2b').set_weights(siamese.get_layer(name='bn3b_branch2b').get_weights())
    siamese.get_layer(name='res30b_branch2c').set_weights(siamese.get_layer(name='res3b_branch2c').get_weights())
    siamese.get_layer(name='bn30b_branch2c').set_weights(siamese.get_layer(name='bn3b_branch2c').get_weights())

    # 3c
    siamese.get_layer(name='res30c_branch2a').set_weights(siamese.get_layer(name='res3c_branch2a').get_weights())
    siamese.get_layer(name='bn30c_branch2a').set_weights(siamese.get_layer(name='bn3c_branch2a').get_weights())
    siamese.get_layer(name='res30c_branch2b').set_weights(siamese.get_layer(name='res3c_branch2b').get_weights())
    siamese.get_layer(name='bn30c_branch2b').set_weights(siamese.get_layer(name='bn3c_branch2b').get_weights())
    siamese.get_layer(name='res30c_branch2c').set_weights(siamese.get_layer(name='res3c_branch2c').get_weights())
    siamese.get_layer(name='bn30c_branch2c').set_weights(siamese.get_layer(name='bn3c_branch2c').get_weights())

    # 3d
    siamese.get_layer(name='res30d_branch2a').set_weights(siamese.get_layer(name='res3d_branch2a').get_weights())
    siamese.get_layer(name='bn30d_branch2a').set_weights(siamese.get_layer(name='bn3d_branch2a').get_weights())
    siamese.get_layer(name='res30d_branch2b').set_weights(siamese.get_layer(name='res3d_branch2b').get_weights())
    siamese.get_layer(name='bn30d_branch2b').set_weights(siamese.get_layer(name='bn3d_branch2b').get_weights())
    siamese.get_layer(name='res30d_branch2c').set_weights(siamese.get_layer(name='res3d_branch2c').get_weights())
    siamese.get_layer(name='bn30d_branch2c').set_weights(siamese.get_layer(name='bn3d_branch2c').get_weights())
    
    # 4a
    siamese.get_layer(name='res40a_branch2a').set_weights(siamese.get_layer(name='res4a_branch2a').get_weights())
    siamese.get_layer(name='bn40a_branch2a').set_weights(siamese.get_layer(name='bn4a_branch2a').get_weights())
    siamese.get_layer(name='res40a_branch2b').set_weights(siamese.get_layer(name='res4a_branch2b').get_weights())
    siamese.get_layer(name='bn40a_branch2b').set_weights(siamese.get_layer(name='bn4a_branch2b').get_weights())
    siamese.get_layer(name='res40a_branch2c').set_weights(siamese.get_layer(name='res4a_branch2c').get_weights())
    siamese.get_layer(name='bn40a_branch2c').set_weights(siamese.get_layer(name='bn4a_branch2c').get_weights())
    siamese.get_layer(name='res40a_branch1').set_weights(siamese.get_layer(name='res4a_branch1').get_weights())
    siamese.get_layer(name='bn40a_branch1').set_weights(siamese.get_layer(name='bn4a_branch1').get_weights())

    # 4b
    siamese.get_layer(name='res40b_branch2a').set_weights(siamese.get_layer(name='res4b_branch2a').get_weights())
    siamese.get_layer(name='bn40b_branch2a').set_weights(siamese.get_layer(name='bn4b_branch2a').get_weights())
    siamese.get_layer(name='res40b_branch2b').set_weights(siamese.get_layer(name='res4b_branch2b').get_weights())
    siamese.get_layer(name='bn40b_branch2b').set_weights(siamese.get_layer(name='bn4b_branch2b').get_weights())
    siamese.get_layer(name='res40b_branch2c').set_weights(siamese.get_layer(name='res4b_branch2c').get_weights())
    siamese.get_layer(name='bn40b_branch2c').set_weights(siamese.get_layer(name='bn4b_branch2c').get_weights())

    # 4c
    siamese.get_layer(name='res40c_branch2a').set_weights(siamese.get_layer(name='res4c_branch2a').get_weights())
    siamese.get_layer(name='bn40c_branch2a').set_weights(siamese.get_layer(name='bn4c_branch2a').get_weights())
    siamese.get_layer(name='res40c_branch2b').set_weights(siamese.get_layer(name='res4c_branch2b').get_weights())
    siamese.get_layer(name='bn40c_branch2b').set_weights(siamese.get_layer(name='bn4c_branch2b').get_weights())
    siamese.get_layer(name='res40c_branch2c').set_weights(siamese.get_layer(name='res4c_branch2c').get_weights())
    siamese.get_layer(name='bn40c_branch2c').set_weights(siamese.get_layer(name='bn4c_branch2c').get_weights())

    # 4d
    siamese.get_layer(name='res40d_branch2a').set_weights(siamese.get_layer(name='res4d_branch2a').get_weights())
    siamese.get_layer(name='bn40d_branch2a').set_weights(siamese.get_layer(name='bn4d_branch2a').get_weights())
    siamese.get_layer(name='res40d_branch2b').set_weights(siamese.get_layer(name='res4d_branch2b').get_weights())
    siamese.get_layer(name='bn40d_branch2b').set_weights(siamese.get_layer(name='bn4d_branch2b').get_weights())
    siamese.get_layer(name='res40d_branch2c').set_weights(siamese.get_layer(name='res4d_branch2c').get_weights())
    siamese.get_layer(name='bn40d_branch2c').set_weights(siamese.get_layer(name='bn4d_branch2c').get_weights())

    # 4e
    siamese.get_layer(name='res40e_branch2a').set_weights(siamese.get_layer(name='res4e_branch2a').get_weights())
    siamese.get_layer(name='bn40e_branch2a').set_weights(siamese.get_layer(name='bn4e_branch2a').get_weights())
    siamese.get_layer(name='res40e_branch2b').set_weights(siamese.get_layer(name='res4e_branch2b').get_weights())
    siamese.get_layer(name='bn40e_branch2b').set_weights(siamese.get_layer(name='bn4e_branch2b').get_weights())
    siamese.get_layer(name='res40e_branch2c').set_weights(siamese.get_layer(name='res4e_branch2c').get_weights())
    siamese.get_layer(name='bn40e_branch2c').set_weights(siamese.get_layer(name='bn4e_branch2c').get_weights())

    # 4f
    siamese.get_layer(name='res40f_branch2a').set_weights(siamese.get_layer(name='res4f_branch2a').get_weights())
    siamese.get_layer(name='bn40f_branch2a').set_weights(siamese.get_layer(name='bn4f_branch2a').get_weights())
    siamese.get_layer(name='res40f_branch2b').set_weights(siamese.get_layer(name='res4f_branch2b').get_weights())
    siamese.get_layer(name='bn40f_branch2b').set_weights(siamese.get_layer(name='bn4f_branch2b').get_weights())
    siamese.get_layer(name='res40f_branch2c').set_weights(siamese.get_layer(name='res4f_branch2c').get_weights())
    siamese.get_layer(name='bn40f_branch2c').set_weights(siamese.get_layer(name='bn4f_branch2c').get_weights())

    # 5a
    siamese.get_layer(name='res50a_branch2a').set_weights(siamese.get_layer(name='res5a_branch2a').get_weights())
    siamese.get_layer(name='bn50a_branch2a').set_weights(siamese.get_layer(name='bn5a_branch2a').get_weights())
    siamese.get_layer(name='res50a_branch2b').set_weights(siamese.get_layer(name='res5a_branch2b').get_weights())
    siamese.get_layer(name='bn50a_branch2b').set_weights(siamese.get_layer(name='bn5a_branch2b').get_weights())
    siamese.get_layer(name='res50a_branch2c').set_weights(siamese.get_layer(name='res5a_branch2c').get_weights())
    siamese.get_layer(name='bn50a_branch2c').set_weights(siamese.get_layer(name='bn5a_branch2c').get_weights())
    siamese.get_layer(name='res50a_branch1').set_weights(siamese.get_layer(name='res5a_branch1').get_weights())
    siamese.get_layer(name='bn50a_branch1').set_weights(siamese.get_layer(name='bn5a_branch1').get_weights())

    # 5b
    siamese.get_layer(name='res50b_branch2a').set_weights(siamese.get_layer(name='res5b_branch2a').get_weights())
    siamese.get_layer(name='bn50b_branch2a').set_weights(siamese.get_layer(name='bn5b_branch2a').get_weights())
    siamese.get_layer(name='res50b_branch2b').set_weights(siamese.get_layer(name='res5b_branch2b').get_weights())
    siamese.get_layer(name='bn50b_branch2b').set_weights(siamese.get_layer(name='bn5b_branch2b').get_weights())
    siamese.get_layer(name='res50b_branch2c').set_weights(siamese.get_layer(name='res5b_branch2c').get_weights())
    siamese.get_layer(name='bn50b_branch2c').set_weights(siamese.get_layer(name='bn5b_branch2c').get_weights())

    # 5c
    siamese.get_layer(name='res50c_branch2a').set_weights(siamese.get_layer(name='res5c_branch2a').get_weights())
    siamese.get_layer(name='bn50c_branch2a').set_weights(siamese.get_layer(name='bn5c_branch2a').get_weights())
    siamese.get_layer(name='res50c_branch2b').set_weights(siamese.get_layer(name='res5c_branch2b').get_weights())
    siamese.get_layer(name='bn50c_branch2b').set_weights(siamese.get_layer(name='bn5c_branch2b').get_weights())
    siamese.get_layer(name='res50c_branch2c').set_weights(siamese.get_layer(name='res5c_branch2c').get_weights())
    siamese.get_layer(name='bn50c_branch2c').set_weights(siamese.get_layer(name='bn5c_branch2c').get_weights())

    siamese.summary()
    return siamese

if __name__ == '__main__':
    self_sup_resnet(shape=(288,512,3))