import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
from tensorflow.keras.layers import BatchNormalization, Input, Flatten, Dense, Lambda, Activation,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras.models import Sequential


"""
## Set up the gradient ascent process
The "loss" we will maximize is simply the mean of the activation of a specific filter in
our target layer. To avoid border effects, we exclude border pixels.
"""


def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


"""
Our gradient ascent function simply computes the gradients of the loss above
with regard to the input image, and update the update image so as to move it
towards a state that will activate the target filter more strongly.
"""


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


"""
## Set up the end-to-end filter visualization loop
Our process is as follow:
- Start from a random image that is close to "all gray" (i.e. visually netural)
- Repeatedly apply the gradient ascent step function defined above
- Convert the resulting input image back to a displayable form, by normalizing it,
center-cropping it, and restricting it to the [0, 255] range.
"""


def initialize_image():
    # We start from a gray image with some random noise
    # img = tf.random.uniform((1, img_width, img_height, 3))
    img = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

def vgg_net(shape=(576,1024, 3)):
    input_1 = Input(shape, name='self_input')
    # vgg16主干网络
    feat1, feat2, feat3, feat4, feat5 = VGG16_1(input_1)
    return  Model(inputs=input_1, outputs=feat5)


def resnet_net(shape=(576,1024, 3)):
    input_1 = Input(shape, name='self_input')
    # vgg16主干网络
    feat1, feat2, feat3, feat4, feat5 = ResNet50(input_1)
    return  Model(inputs=input_1, outputs=feat5)

from nets.vgg16 import VGG16 as VGG16_1
# from keras.applications.vgg16 import VGG16
from nets.resnet50 import ResNet50

if __name__ == '__main__':
    # The dimensions of our input image
    img_width = 180
    img_height = 180
    # Our target layer: we will visualize the filters from this layer.
    # See `model.summary()` for list of layer names, if you want to change this.
    # layer_name = " block5_pool"
    layer_name = "bn4c_branch2c"

    """
    ## Build a feature extraction model
    """

    # Build a ResNet50V2 model loaded with pre-trained ImageNet weights
    # model = keras.models.load_model('/tmp/pycharm_project_116/self_supervised2/self_logs/h5_2022_03_19_13_13_15/ep005-loss0.041-val_loss0.093.h5')
    # model = VGG16(weights="imagenet", include_top=False)
    # model.summary()

    model = resnet_net(shape=(180, 180, 3))
    model.summary()
    # model.load_weights("/tmp/pycharm_project_116/self_supervised2/self_logs/h5_2022_03_19_12_31_46/ep008-loss0.028-val_loss0.334.h5",by_name=True)
    model.load_weights(
        "/home/dell/python/Cherry_unsup/self_supervised_resnet/self_logs/h5_2022_04_18_21_12_37_对比分类100000/ep010-loss0.045-val_loss0.050.h5",


        by_name=True)

    # Set up a model that returns the activation values for our target layer
    layer = model.get_layer(name=layer_name)
    feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    feature_extractor.summary()

    loss, img = visualize_filter(0)
    keras.preprocessing.image.save_img("0.png", img)

    """
    This is what an input that maximizes the response of filter 0 in the target layer would
    look like:
    """

    display(Image("0.png"))

    """
    ## Visualize the first 64 filters in the target layer
    Now, let's make a 8x8 grid of the first 64 filters
    in the target layer to get of feel for the range
    of different visual patterns that the model has learned.
    """

    # Compute image inputs that maximize per-filter activations
    # for the first 64 filters of our target layer
    all_imgs = []
    for filter_index in range(81):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index)
        all_imgs.append(img)

    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = 9
    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img = all_imgs[i * n + j]
            stitched_filters[
            (cropped_width + margin) * i: (cropped_width + margin) * i + cropped_width,
            (cropped_height + margin) * j: (cropped_height + margin) * j
                                           + cropped_height,
            :,
            ] = img
    keras.preprocessing.image.save_img("stiched_filters强数据增强.png", stitched_filters)

    # from IPython.display import Image, display

    # display(Image("stiched_filters中数据增强.png"))

