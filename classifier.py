import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os, ssl, time
import PIL.ImageOps
from PIL import Image 

if (not os.environ.get("PYTHONHTTPSVERIFY",'') and getattr(ssl,"_create_unverified_context",None)):
    ssl._create_default_https_context =ssl._create_unverified_context
x,y= fetch_openml('mnist_784', version=1, return_X_y= True)
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=9,train_size=7500, test_size=2500)
x_train_scale=x_train/255.0
x_test_scale=x_test/255.0
clf= LogisticRegression(solver="saga", multi_class= "multinomial")
clf.fit(x_train_scale, y_train)

def get_prediction(img):
    im_pil=Image.open(img)
    image_bw=im_pil.convert('L')
    image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)
    image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)
    pixel_filter=20
    minimum_pixel=np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-minimum_pixel,0,255)
    maxpixel=np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/maxpixel
    test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred=clf.predict(test_sample)
    return test_pred[0]

