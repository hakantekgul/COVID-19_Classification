import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

print('Please enter the name of the image for testing COVID 19:')
user_input = input()
loaded_model = tf.keras.models.load_model('covid_model.h5')
to_predict = cv2.imread(str(user_input))
imge = cv2.cvtColor(to_predict,cv2.COLOR_BGR2RGB)
resized = cv2.resize(imge,(224,224), interpolation = cv2.INTER_AREA)
pred_img = np.array(resized) / 255.0
pred_img = pred_img.reshape((1,224,224,3))
preds = loaded_model.predict(pred_img,batch_size=1)
pred = np.argmax(preds, axis=1)

os.system('clear')
if pred == 1:
    result = 'Positive'
    print('THE RESULT IS POSITIVE')
else:
    result = 'Negative'
plt.imshow(to_predict,'gray')
plt.title('Result: '+ str(result))
plt.axis('off')
plt.show()