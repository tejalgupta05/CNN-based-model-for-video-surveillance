import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


model = tf.keras.models.load_model(r"D:\ML\Mini Project CNN\secondModel(on clg images).h5")
model.summary()

img = cv2.imread(r"D:\ML\Mini Project CNN\Clg Images\0\Frame 1371.jpg")
img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)

# Add dim 0
img = np.expand_dims(img, axis=0)

outputs = [layer.output for layer in model.layers]

visualisation_model = tf.keras.models.Model(inputs=model.input, outputs=outputs)
predictions_by_layers = visualisation_model.predict(img)

print(model.predict(img))

layer_names = [layer.name for layer in model.layers]

for layer_name, predictions_by_layer in zip(layer_names, predictions_by_layers):
    if len(predictions_by_layer.shape) == 4:
        no_features = predictions_by_layer.shape[-1]
        size = predictions_by_layer.shape[1]

        grid = np.zeros((size, size*no_features))
        for i in range(no_features):
            x = predictions_by_layer[0, :, :, i]
            
            grid[:, i*size:(i+1)*size] = x

        scale = 20/no_features
        plt.figure(figsize=(scale*no_features, scale))
        plt.title(layer_name)
        plt.imshow(grid)
        plt.show()
