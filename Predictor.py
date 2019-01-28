# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:08:29 2018

@author: vewald
"""

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


img_path = 'data/SPIE2019/All/test/Sample2.png' # Relative path of an image in the Test folder which you would like to test
model_path = 'predictor.h5' # The path of the saved (trained) model
classlabelmsgs = ['a = 0 mm','a = 15 mm','a = 30 mm','a = 45 mm','a = 60 mm','a = 75 mm']
classlabels = ['0 mm','15 mm','30 mm','45 mm','60 mm','75 mm']

model = load_model(model_path)

def img_transformer(img_path):
    """
    This function is meant for performing the operation of loading the image then preprocessing it in the same manner as it was
    done for each and every one of the images which were supplied to the FinalCNN.py script for training
    :param img_path: The relative path of an image you wish to test out
    :return: the preprocessed image
    """
    img_width, img_height = 5001, 108
    img = image.load_img((img_path), target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    myimage = np.vstack([x])
    return myimage

class_probs = model.predict(img_transformer(img_path))
class_probs *= 100

def pred_class_label(class_probs):
    """
    This function is simply meant for facilitating the job of extracting the bin with the highest value
    within the probability class array
    :param class_probs: probability class array
    :return: The class label and class label message corresponding with the bin (element) in the probability class array
    with the highest value
    """
    if class_probs.max() == class_probs[0][0]:
        return classlabelmsgs[0],classlabels[0]
    elif class_probs.max() == class_probs[0][1]:
        return classlabelmsgs[1],classlabels[1]
    elif class_probs.max() == class_probs[0][2]:
        return classlabelmsgs[2],classlabels[2]
    elif class_probs.max() == class_probs[0][3]:
        return classlabelmsgs[3],classlabels[3]
    elif class_probs.max() == class_probs[0][4]:
        return classlabelmsgs[4],classlabels[4]        
    else:
        return classlabelmsgs[5],classlabels[5]

def true_class_label(img_path):
    """
    This function is simply meant for extracting the true class label of the test image
    :param img_path: The relative path of an image you wish to test out
    :return: The true class label of the image
    """
    for i in img_path.split('/'):
        if i in classlabels:
            return i

classes = model.predict_classes(img_transformer(img_path))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\nArray of prediction probabilities:",class_probs)
print('\nLargest prediction probabilities array element message: This image belongs to bin/element number', classes, 'in the prob class array above')
d00_prob_str = str(np.around(class_probs[0][0], decimals = 1)) +'%'
d15_prob_str = str(np.around(class_probs[0][1], decimals = 1)) +'%'
d30_prob_str = str(np.around(class_probs[0][2], decimals = 1)) +'%'
d45_prob_str = str(np.around(class_probs[0][3], decimals = 1)) +'%'
d60_prob_str = str(np.around(class_probs[0][4], decimals = 1)) +'%'
d75_prob_str = str(np.around(class_probs[0][5], decimals = 1)) +'%'
prediction_msg = pred_class_label(class_probs)
print("\nPrediction message:", prediction_msg[0])
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print('The percentage probabilities computed for each possible label/class are:\n')
print('a = 00 mm:', d00_prob_str)
print('a = 15 mm:', d15_prob_str)
print('a = 30 mm:', d30_prob_str)
print('a = 45 mm:', d45_prob_str)
print('a = 60 mm:', d60_prob_str)
print('a = 75 mm:', d75_prob_str)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

img = cv2.imread(img_path)
#textboxstr = "a = 00 mm:{0}\na = 15 mm:{1}\na = 30 mm:{2}\na = 45 mm:{3}\na = 60 mm:{4}\na = 75 mm:{5}\n".format(d00_prob_str, d15_prob_str,
#                                                                                       d30_prob_str, d45_prob_str,
#                                                                                       d60_prob_str, d75_prob_str)
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.5)
ax.set_title("Predicted Crack Classification Label: {0}".format(pred_class_label(class_probs)[1]))
#ax.set_title("""\nTrue Crack Classification Label: {0}
#\nPredicted Crack Classification Label: {1}""".format(true_class_label(img_path),pred_class_label(class_probs)[1]))
#ax.text(0.95, 0.01, textboxstr,
#        verticalalignment='bottom', horizontalalignment='right',
#        transform=ax.transAxes,
#        color='green', fontsize=15)
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()