import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import pandas as pd

df = pd.read_csv('dataset/cat_dog.csv')
print('Tổng số ảnh: ', len(df))

X=[]
y=[]

for index, row in df.iterrows():
    img_path = f'dataset/cat_dog/{row['image']}'
    img = Image.open(img_path).resize((64,64))
    X.append(np.array(img))
    y.append(row['labels'])
print('Đọc ảnh xong')
# Bước 3: Chuẩn bị data cho CNN

X=np.array(X)
X = X/255.0
y=np.array(y)
print("Shape của X:", X.shape)
print("Shape của y:", y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test= train_test_split(X,y, test_size=0.2, random_state=42)
print('Số ảnh train', len(X_train))
print("Số ảnh test:", len(X_test))

# Bước 5: Xây CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(64,64,3) ),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation = 'relu', input_shape=(64,64,3) ),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation = 'relu', input_shape=(64,64,3) ),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# Bước 6: Train CNN
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)
# Bước 7: Lưu model CNN
model.save("model_cnn.h5")
print("Đã lưu model CNN!")