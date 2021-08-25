import numpy as np
import pandas as pd
#libraraies to create prediction model
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

#there are images given we are predicting the images using the postman API
# open ml ia a open source where you can share or fetch dataset
#create classifier which takes data from openml,version =1 and fetching x and y value
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

#split the data into train and test data
#random_state is basically used for reproducing your problem the same every time it is run. 

#If you do not use a random_state in train_test_split, 
# every time you make the split you might get a different set of train and test data points
# and will not help you in debugging in case you get an issue.
# This is to check and validate the data when running the code multiple times. 
# Setting random_state a fixed value will guarantee that same sequence of random numbers are generated each time you run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

#Now we need to scale the data to make sure that the data points in X and Y are equal 
# so we will divide them using 255 which is the maximum pixel of the image
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Solver Saga is algorithm to use in the optimisation problem so for big data said 
# Solver Saga is good but when you have a small datasets then you can use live liblinear
#For multinomial the loss  minimised is the multinomial loss fit across the entire property distribution even when the data is binary 
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

#If you remember what we did earlier we were using CV2  
# and using CV2 we were using a device’s camera and capturing each frame.
# Now each frame was an image where we were doing some processing and then predicting the value from it.
# For the same thing we will create a function to do that

def get_prediction(image):
    im_pil = Image.open(image)
    #We are converting our image into a scalar quantity and 
    # then we are converting into greyscale image so that colour don't interfere with prediction
    image_bw = im_pil.convert('L')
    #We have resized the image into (28,28) using Image.ANTIALIAS
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    #Then using the percentile function get the minimum pixel 
    min_pixel = np.percentile(image_bw_resized, pixel_filter)

    #https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    #Given an interval, values outside the interval are clipped to the interval edges. 
    #For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
    #we are converting image into a number
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
     # getting the max pixel
    max_pixel = np.max(image_bw_resized)

    # we are creating an array
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    #Gives a new shape to an array without changing its data.(28X28)
    
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0] 
