from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Bước 1: Đọc dataset
X = []
y = []

for label in ["cat", "dog"]:
    folder = f"dataset/{label}"
    for filename in os.listdir(folder):
        img = Image.open(f"{folder}/{filename}").resize((64, 64))
        X.append(np.array(img))
        y.append(label)

print("Số ảnh:", len(X))
print("Labels:", y)

# Bước 2: Chuẩn bị data cho model
X = np.array(X)                        # Chuyển list thành numpy array
X = X.reshape(len(X), -1)             # Flatten: (6, 64, 64, 3) → (6, 12288)
                                        # -1 = "tao không cần biết số cột, các hàng, còn lại tự tính đi"
print("Shape của X sau reshape:", X.shape)
print("Shape của 1 ảnh:", X[0].shape)

# Bước 3: Chia dataset thành train và test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Số ảnh train:", len(X_train))
print("Số ảnh test:", len(X_test))
# Bước 4: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


print("Train xong!")

# Bước 5: Kiểm tra độ chính xác
y_pred = model.predict(X_test)
print("Dự đoán:", y_pred)
print("Thực tế:", y_test)
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)
# Bước 6: Lưu model
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Đã lưu model vào model.pkl!")

# Test load lại
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

print("Load model thành công!")
print("Test dự đoán:", loaded_model.predict([X_test[0]]))
# Bước 6: Lưu model
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Đã lưu model vào model.pkl!")

# Test load lại
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

print("Load model thành công!")
print("Test dự đoán:", loaded_model.predict([X_test[0]]))