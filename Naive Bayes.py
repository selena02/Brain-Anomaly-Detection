import numpy as np
import cv2 as cv2
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix

imagini_antrenare = []
clase_antrenare = []

# citirea datelor de antrenare
with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', 'r') as train:
    next(train)
    for line in train:
        line = line.strip()
        id = line.split(',')[0]
        clasa = int(line.split(',')[1])
        img = cv2.imread(f"/kaggle/input/unibuc-brain-ad/data/data/{id}.png")
        imagini_antrenare.append(img)
        clase_antrenare.append(int(clasa))

imagini_antrenare = np.array(imagini_antrenare)

imagini_antrenare = imagini_antrenare.reshape((imagini_antrenare.shape[0], -1))
#definirea si antrenarea modelului
model = GaussianNB()
model.fit(imagini_antrenare, clase_antrenare)

imagini_validare = []
clase_validare = []
with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', 'r') as validation:
        next(validation)
        for line in validation:
                line = line.strip()
                id = line.split(',')[0]
                clasa = int(line.split(',')[1])
                img = cv2.imread(f"/kaggle/input/unibuc-brain-ad/data/data/{id}.png")
                imagini_validare.append(img)
                clase_validare.append(int(clasa))
imagini_validare = np.array(imagini_validare)
imagini_validare = imagini_validare.reshape((imagini_validare.shape[0], -1))

predictie = model.predict(imagini_validare)
f1 = f1_score(predictie, clase_validare )
print(f"F1 scor: {f1}\n")

imagini_testare = []
for i in range(17001,22150):
    id = str(0) + str(i)
    img = cv2.imread(f"/kaggle/input/unibuc-brain-ad/data/data/{id}.png")
    imagini_testare.append(img)

    imagini_testare = np.array(imagini_testare)
    imagini_testare = imagini_testare.reshape((imagini_testare.shape[0], -1))
    predictie_finala = model.predict(imagini_testare)

j = 17001
with open('final_prediction.csv', 'a') as f:
    f.write('id,class')
    f.write('\n')
    for i in range(len(predictie_finala)):
        f.write(f'0{j}')
        f.write(',')
        f.write(str(predictie_finala[i]))
        f.write('\n')
        j += 1

f.close()    