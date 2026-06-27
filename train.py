from PIL import Image
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Bước 1: Đọc file CSV chứa nhãn
df = pd.read_csv("dataset/cat_dog.csv")
print("Tổng số ảnh:", len(df))
print(df.head())

# Bước 2: Đọc ảnh từ folder dựa theo CSV
X = []
y = []

for index, row in df.iterrows():
    img_path = f'dataset/cat_dog/{row['image']}'
    img = Image.open(img_path).resize((64,64))
    X.append(np.array(img))
    y.append(row['labels'])
print('Done reading')
print('Number of pics: ', len(X))

X= np.array(X)
X= X.reshape(len(X), -1)
print("Shape của X:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print('Đang train')
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print('Train xong')
y_pred = model.predict(X_test)
print("Độ chính xác: ", accuracy_score(y_test, y_pred))

#Lưu model vào file
import pickle

with open ('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Đã lưu model')
