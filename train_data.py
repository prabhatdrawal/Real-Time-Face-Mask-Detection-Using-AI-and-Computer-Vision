from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# intialize the initial learning rate , number of epochs to train for and batch size 
initial_learning_rate =1 * 10**-4

epochs = 20
bs = 32

directory = r"/Users/prabhatrawal/Desktop/MASK_DETECTION/dataset"
categories = ["with_mask","without_mask"]

#grab the list of images in our directory then intialize the list of data
# images and class images
print("[info]loading images....")

data = []
labels = []

for category in categories:
    path = os.path.join(directory, category) # access the file in the directory of the given category
    for img in os.listdir(path): # list down all the images in the particular directory
        img_path = os.path.join(path,img)
        image = load_img(img_path,target_size = (224,224)) # load_img include in keras.preprocessing.image loads image
        #target size is height and width of the image
        image = img_to_array(image) # converts image to array // same lib as load_img
        image = preprocess_input(image) # for the usage of mobilenets

        data.append(image)
        labels.append(category)

# perform one hot encoding on the labels
lb = LabelBinarizer() # converting into categorical variable
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data,dtype = "float32")
labels = np.array(labels)



# out of all the images in the directory 20% is given to test by test_size
(trainX, testX, trainY, testY) = train_test_split(data,labels,
                                                  test_size=0.20, stratify=labels,
                                                    random_state=42)

#construct the training image generator for data augmentation
aug  = ImageDataGenerator(
    rotation_range =20,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest"
)

#load the MobileNetV2 network, ensuring the head fc layer sets are
#left off
baseModel = MobileNetV2(weights="imagenet",include_top = False,
                       input_tensor=Input(shape=(224,224,3)) )

headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,activation = "softmax")(headModel)
 
#the actual mode will be train here 
#Model is used for train data where the parameter is takes are 
#input and output only
model = Model(inputs = baseModel.input,outputs = headModel)

for layer in baseModel.layers:
    layer.trainable = False

#block to compile the  model
print("[info]compiling the model...")
opt = Adam(learning_rate=initial_learning_rate,decay = initial_learning_rate/epochs)
model.compile(loss="binary_crossentropy",optimizer = opt, metrics = ["accuracy"])

print("[info] training head")
H = model.fit(
    aug.flow(trainX,trainY,batch_size=bs),
    steps_per_epoch=len(trainX) // bs,
    validation_data=(testX,testY),
    validation_steps=len(testX)//bs,
    epochs = epochs)

print("[info] evaluating network")
preIdxs = model.predict(testX, batch_size=bs) # predict is used to access
#prediction in the model
# Convert predicted probabilities to class labels
preIdxs = np.argmax(preIdxs, axis=1)  

# Print classification report
print(classification_report(testY.argmax(axis=1), preIdxs, target_names=lb.classes_))

#saving the model to the disk
print("[info] saving mask detector model")
model.save("mask_detector.h5")  # Just specify the filename with .h5 extension


n=epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,n),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,n),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,n),H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,n),H.history["val_accuracy"],label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("loss/accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
