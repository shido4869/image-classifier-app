

# 1. List - danh sách
animals = ["cat", "dog", "bird"]
print(animals[0])        # Lấy phần tử đầu tiên
print(len(animals))      # Đếm số phần tử

# 2. Loop - vòng lặp
for animal in animals:
    print("This is a:", animal)

# 3. Function - hàm
def greet(name):
    return "Hello " + name

print(greet("Manh"))

image_info = {
    "filename": "cat.jpg",
    "label" : "cat",
    "size": 123
}

print(image_info['label'])

def check(label) :
    if label == "cat":
        return "Ok cat"
    else :
        return "Ok dog"

print(check(image_info['label']))


# 6. Dictionary + Loop kết hợp
dataset = [
    {"name": "human1", "job": "hunter"},
    {"name" : "human2", "job" : "guild master"},
    {'name' : 'human3', 'job' : 'mage'}
]

for i in dataset:
    print(i['name'] , 'and', i['job'])

#=============================================================

# 7. Đọc file ảnh bằng Pillow
from PIL import Image

# Tạo thử 1 ảnh giả để test
img = Image.new("RGB", (224, 224), color=(255, 0, 0))
img.save("test_image.jpg")
print("Tạo ảnh xong!")

# Mở lại ảnh vừa tạo
img = Image.open("test_image.jpg")
print("Kích thước ảnh:", img.size)
print("Mode:", img.mode)

# Resize ảnh về 64x64
img_resized = img.resize((64, 64))
print("Sau khi resize:", img_resized.size)

# Chuyển ảnh thành array số
import numpy as np
img_array = np.array(img_resized)
print("Shape của array:", img_array.shape)
print("Giá trị pixel đầu tiên:", img_array[0][0])

#===========================================

from PIL import Image
import numpy as np
import os

# Tạo giả dataset: 2 folder cat/ và dog/
os.makedirs("dataset/cat", exist_ok=True)
os.makedirs("dataset/dog", exist_ok=True)

# Tạo vài ảnh giả
for i in range(3):
    Image.new("RGB", (224, 224), color=(255, 0, 0)).save(f"dataset/cat/cat{i}.jpg")
    Image.new("RGB", (224, 224), color=(0, 0, 255)).save(f"dataset/dog/dog{i}.jpg")

print("Đã tạo dataset giả!")

# Đọc toàn bộ dataset
X = []  # chứa array ảnh
y = []  # chứa nhãn (label)

for label in ["cat", "dog"]:
    folder = f"dataset/{label}"
    for filename in os.listdir(folder):
        img = Image.open(f"{folder}/{filename}").resize((64, 64))
        X.append(np.array(img))
        y.append(label)

#  X = [array_cat0, array_cat1, array_cat2, array_dog0, array_dog1, array_dog2]
# y = ["cat",      "cat",      "cat",      "dog",      "dog",      "dog"]


print("Số ảnh đọc được:", len(X))
print("Labels:", y)
print("Shape 1 ảnh:", X[0].shape)
#========================================
