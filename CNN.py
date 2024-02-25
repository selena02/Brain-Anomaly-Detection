import pandas as pd
import numpy as np
import cv2


import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from sklearn.utils.class_weight import compute_class_weight

# lista pentru stocarea claselor de antrenare si path-urile catre imaginile de antrenare
clase_antrenare = []
antrenare_path = []

with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', 'r') as train:
    next(train)
    for line in train:
        line = line.strip()
        id = line.split(',')[0]
        clasa = line.split(',')[1]
        img_path = f"/kaggle/input/unibuc-brain-ad/data/data/{id}.png"  
        clase_antrenare.append(clasa)
        antrenare_path.append(img_path)  

# transformarea claselor de antrenare in intregi
clase_antrenare_int = [int(label) for label in clase_antrenare]

# calcularea ponderilor de clasa, in functie de distributia claselor din setul de antrenare

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(clase_antrenare_int),
                                     y=clase_antrenare_int)

dictionar = dict(enumerate(class_weights))

antrenare = pd.DataFrame({'image': antrenare_path, 'label': clase_antrenare})

# citirea path si clase pentru validare
clase_validare = []
validare_path = []
with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', 'r') as validation:
    next(validation)
    for line in validation:
        line = line.strip()
        id = line.split(',')[0]
        clasa = line.split(',')[1]
        img_path = f"/kaggle/input/unibuc-brain-ad/data/data/{id}.png"  # get the file path of the image
        validare_path.append(img_path)
        clase_validare.append(clasa)

validare = pd.DataFrame({'image': validare_path, 'label': clase_validare})

antrenareGenerator = ImageDataGenerator(rescale=1. / 255.)
validareGenerator = ImageDataGenerator(rescale=1. / 255.)

antrenareDataset = antrenareGenerator.flow_from_dataframe(
    dataframe=antrenare,
    class_mode="binary",
    shuffle=False,
    x_col="image",
    y_col="label",
    batch_size=32,
    color_mode="grayscale",
    target_size=(224, 224)
)
validareDataset = validareGenerator.flow_from_dataframe(
    dataframe=validare,
    class_mode='binary',
    shuffle=False,
    x_col="image",
    y_col="label",
    batch_size=32,
    color_mode="grayscale",

    target_size=(224, 224)
)
# definirea modelului
model = keras.Sequential([
    
    keras.layers.InputLayer(input_shape=(224, 224, 1)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(512, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(antrenareDataset, epochs=20, validation_data=validareDataset, class_weight=dictionar)

test_path = []  

for i in range(17001, 22150):
    id = str(0) + str(i)
    path = f"/kaggle/input/unibuc-brain-ad/data/data/{id}.png"
    test_path.append(path)

test = pd.DataFrame({'image': test_path})

testGenerator = ImageDataGenerator(rescale=1./255.)
testDataset = testGenerator.flow_from_dataframe(
  dataframe = test,
  class_mode = None,
  shuffle=False,
  x_col = "image",
  y_col = None,
  batch_size = 32,
  color_mode="grayscale",
  target_size=(224, 224)
)

predictions = model.predict(testDataset)
final_pred = np.where(predictions > 0.5, 1, 0)
j = 17001
with open('cnn_prediction.csv', 'w') as f:
    f.write('id,class')
    f.write('\n')
    for i in range(len(final_pred)):
        f.write(f'0{j}')
        f.write(',')
        f.write(str(final_pred[i]).strip('[]'))
        f.write('\n')
        j += 1

f.close() 