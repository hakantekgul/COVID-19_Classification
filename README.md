# COVID-19_Classification
A classifier to detect whether a patient has COVID 19 virus or not from chest X-Ray images.

The model is already trained and ready to test for chest X-Ray images. In order to predict a single image, run the following command;

```sh
python3 predict.py
```

After that, simply enter the path of the X-Ray image. You should see the image itself with the result of prediction. 

--> The jupyter notebook given above trains the model with Keras and Tensorflow. COVID-19 positive images are taken from an open source dataset, and healthy images are taken from a Kaggle dataset. 


- COVID-19 Images: https://github.com/ieee8023/covid-chestxray-dataset
- Healthy Images: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia 
