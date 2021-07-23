import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2


species = ['Amblyomma', 'Dermacentor', 'Ixodes']

model = keras.models.load_model('my_model')

img_path = 'image.png'

img = Image.open(img_path)
img_resized = img.resize((224, 224))
pixels = np.asarray(img_resized)  # convert image to array
pixels = pixels.astype('float32')

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

