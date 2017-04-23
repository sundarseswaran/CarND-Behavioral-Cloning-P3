import csv
import cv2
import numpy as np
import pandas as pd
import sklearn

from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda

from sklearn.model_selection import train_test_split


#
# Parameter Definition
#
basepath = '../data/'
# basepath = '/Users/sundar/Desktop/Recording/'
crop_top_pixels = 64
crop_bottom_pixels = 20
batch_size = 32
pool_size = (2,2)


#
# Function Definition
#
def getImage(row,pick_camera):
    """Get image from file

    Retrieve the image from the corresponding path. The generator calls this
    function to get a random image for the given row and camera view
    """
    source_path = row.get_value(headers[pick_camera])
    filename = source_path.split('/')[-1]
    file_path = basepath+'IMG/' + filename
    # print('image path => ', file_path)
    img = cv2.imread(file_path)

    # do corrections on measurement based on the camera position on the car.
    # left camera (+0.25), right camera (-0.25)
    camera_correction = {0: 0.0 , 1: 0.25, 2: -0.25}
    measurement = float(row.get_value(headers[3])) + camera_correction[pick_camera]

    # return the image with corrected measurement
    return img, measurement


def imageTranslation(img, measurement, dX, dY, pxSteer):
    """Image translation

    This function translates the image slightly-off from the axis. Used during
    preprocessing or image augmentation steps.
    """

    # get image shape (rows, columns, depth)
    r,c,d = img.shape
    # translate for X & Y
    deltaX = dX * np.random.uniform(-1,1)
    deltaY = dY * np.random.uniform(-1,1)
    steerAdj = measurement + deltaX * pxSteer

    # translation matrix
    M = np.float32([[1,0,deltaX],[0,1,deltaY]])
    trIMG = cv2.warpAffine(img, M, (c, r))

    return trIMG,steerAdj


def imageShadowing(image):

    # converting the image to HSV scale and to type float32 for easy multiplication
    img1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    img1 = img1.astype('float32')
    intensity = img1[:,:,2]

    # Lets get random intensity between 0.3 - 1.3 times the actual intensity.
    # Can go brighter, but making an assumption that the new images will be darker than originals
    new_intensity = intensity * np.random.uniform(0.3,1.3)
    img2 = img1
    img2[:,:,2] = new_intensity

    # convert the image back to RGB Scale
    img2 = img2.astype('uint8')
    img2 = cv2.cvtColor(img2,cv2.COLOR_HSV2RGB)
    return img2

def flipImage(image,measurement):
    """ Flip image left to right

    """

    return np.fliplr(image), -1.0 * measurement


def buildModel():
    """ Build CNN based on Keras 2 API

    Returns a Sequential models built using Keras 2 API. The model is entirely
    based on NVIDIA's end-to-end self-driving-car model
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

    Added Dropouts on the fully connected layer to mitigate overfitting
    """

    model = Sequential()

    model.add(Cropping2D(cropping=((crop_top_pixels, crop_bottom_pixels), (1, 1)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))
    return model

def generator(samples,batch_size=32):
    no_of_samples = len(samples)

    pick_camera = [0,1,2]

    while True:
        ## Randomize the order in which samples are retrived
        sklearn.utils.shuffle(samples)
        for rangeVal in range(0, no_of_samples, batch_size):
            BATCH = samples[rangeVal:rangeVal+batch_size]

            images = []
            measurements = []
            for sample in BATCH:
                # for every sample get the corresponding row
                row = dataset.iloc[sample]
                # retrive the image with random camera position
                image, measurement = getImage(row, np.random.choice(pick_camera,1)[0])# print(image.shape)

                #
                # Perform image augmentation steps
                #
                new_image = image
                new_measurement = measurement

                # call flipImage
                if (np.random.randint(0,2)==0):
                    new_image,new_measurement = flipImage(new_image,new_measurement)
                # do shadowing / brightness adjustments
                new_image = imageShadowing(new_image)
                # perform image translation
                new_image,new_measurement = imageTranslation(new_image,new_measurement,50,20,0.004)
                images.append(new_image)
                measurements.append(new_measurement)

            batchX = np.array(images)
            batchY = np.array(measurements)

            output = sklearn.utils.shuffle(batchX, batchY)
            yield output
######## ends function definitions


#
# Bootstrap model.py
#
csvfile = basepath + 'driving_log.csv'
print('Loading data from CSV file: ', csvfile)

dataset = pd.read_csv(csvfile)
headers = dataset.columns
images=[]
measurements = []

idx_vector = np.arange(0,dataset.shape[0])
train_samples, validation_samples = train_test_split(idx_vector, test_size=0.2)

## Double the samples to gain more training data using randomized preprocessors
no_train_samples = len(train_samples) * 2
no_validation_samples = len(validation_samples) * 2

print('Number of training samples :' , no_train_samples)
print('Number of validation samples :' ,  no_validation_samples)

# instatiate Sequential model
model = buildModel()
model.summary()

training_generator = generator(train_samples,batch_size)
validation_generator = generator(validation_samples,batch_size)

# use MSE loss function, adam - optimizer with default learning rate 0.001
model.compile(optimizer='adam', loss='mse')

# use fit_generator for processing in batches
history = model.fit_generator(training_generator,
                    steps_per_epoch  = no_train_samples,
                    validation_data  = validation_generator,
                    validation_steps = no_validation_samples,
                    epochs=5)

model.save('model.h5')

print('Model Saved')
