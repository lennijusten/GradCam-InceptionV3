import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2


species = ['Amblyomma', 'Dermacentor', 'Ixodes']

model = keras.models.load_model('/Users/Lenni/Downloads/my_model')

img_path = '/Users/Lenni/Desktop/Screen Shot 2020-10-11 at 5.33.57 PM.png'
# base_model = InceptionV3(
#         input_shape=(224,224,3),
#         include_top=False)
# x = base_model.output
# x1 = tf.keras.layers.GlobalMaxPooling2D()(x)
# x2 = tf.keras.layers.Dense(3, activation='softmax')(x1)
# model = tf.keras.models.Model(inputs=base_model.input, outputs=x2)
#
# model.load_weights('/Users/Lenni/Downloads/model_0.003.h5')

# def loadModel():  # Load neural network with saved weights and architecture
#     json_path =  '/Users/Lenni/Downloads/model_0.003.json' # look for any file with .json extension
#     weights_path = '/Users/Lenni/Downloads/model_0.003.h5'  # look for any file with .h5 extension
#
#
#     json_file = open(json_path, 'r')  # load first model instance
#     loaded_model_json = json_file.read()
#     json_file.close()
#
#     loaded_model = model_from_json(loaded_model_json)
#     loaded_model.load_weights(weights_path)  # load first weights instance
#
#     print("Model loaded from disk.")
#     # logging.info("loadModel() -- Success")
#     return loaded_model
#
# model = loadModel()

means = [0.485, 0.456, 0.406]
std = 0.225

img = Image.open(img_path)
img_resized = img.resize((224, 224))
pixels = np.asarray(img_resized)  # convert image to array
pixels = pixels.astype('float32')
# pixels /= 255.0
# pixels = (pixels - means)/std
input = np.expand_dims(pixels, axis=0)

preds = model.predict(input)
print(species[np.argmax(preds[0])])

with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_93')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(input)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)


heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap = heatmap.reshape((5, 5))
plt.matshow(heatmap)
plt.show()

INTENSITY = 0.5

raw = Image.open(img_path)
raw = raw.resize((224,224))
heatmap = cv2.resize(heatmap, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)


img = (heatmap * INTENSITY + np.array(raw))/255

plt.imshow(raw)
plt.show()
plt.imshow(img)
plt.show()

