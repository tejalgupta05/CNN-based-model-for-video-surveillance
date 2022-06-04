from tensorflow import keras
import numpy as np
import cv2


model = keras.models.load_model(r"D:\ML\Mini Project CNN\secondModel(on clg images).h5")
#model.summary()

img = cv2.imread(r"D:\ML\Mini Project CNN\Clg Images\1\Frame 271.jpg")
img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
img = np.expand_dims(img, axis=0)


pred = model.predict(img)
print(pred)

img = cv2.imread(r"D:\ML\Mini Project CNN\Clg Images\1\Frame 271.jpg")
img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)

cv2.putText(img, str(np.argmax(pred)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
cv2.imshow("Image", img)
cv2.waitKey()
