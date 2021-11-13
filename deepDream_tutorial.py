import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL.Image
from tensorflow.keras.preprocessing import image

# following https://www.tensorflow.org/tutorials/generative/deepdream

url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'


# download image and convert to Numpy array
def download(url, max_dim=None):
    # get the name of the image by splitting at last /
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


# normalise the image
def deprocess(img):
    img = 255*(img+1.0)/2.0
    return tf.cast(img, tf.uint8)


# display the image
def show(img):
    plt.imshow(img)
    plt.show()


# downsize the image
original_img = download(url, max_dim=500)
img = deprocess(original_img)
show(img)

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# choose which layers to focus on
# deeper layers respond to higher-level features
# earlier layers respond to simpler features
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

# create feature extraction model
dream_model = tf.keras.Model(inputs)