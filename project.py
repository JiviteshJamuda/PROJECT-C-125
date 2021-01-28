import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

# fetching data
X = np.load('image.npz')['arr_0']
y = pd.read_csv('data.csv')['labels']
# classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# n_classes = len(classes)

# Training the data
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 3500, test_size = 500, random_state = 9)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train_scaled, y_train)


def get_pred(image):
    im_pil = Image.open(image)
    im_bw = im_pil.convert('L')
    im_bw_resized = im_bw.resize((28,28), Image.ANTIALIAS)
    
    pixel_filter = 20
    min_pixel = np.percentile(im_bw_resized, pixel_filter)
    im_scaled = np.clip(im_bw_resized - min_pixel, 0, 255)
    max_pixel = np.max(im_bw_resized)
    im_scaled = np.asarray(im_scaled)/max_pixel
    
    test_sample = np.array(im_scaled).reshape(1,660)
    test_pred = clf.predict(test_sample)
    return test_pred[0]