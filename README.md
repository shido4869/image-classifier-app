\# Cat vs Dog Classifier



A full-stack machine learning web app that classifies images as cats or dogs, built to compare a traditional ML approach against a deep learning approach.



!\[App Screenshot](screenshot.png)



\## Overview



This project walks through the complete ML pipeline: from raw image data to a deployed web interface. Two models were trained and compared to demonstrate the impact of model architecture on performance.



| Model | Accuracy | Approach |

|-------|----------|----------|

| Random Forest | 64.86% | Classical ML on flattened pixel arrays |

| CNN (Convolutional Neural Network) | 83.48% | Deep learning with learned spatial features |



\## Tech Stack



\- \*\*Data processing:\*\* NumPy, Pandas, Pillow

\- \*\*Machine Learning:\*\* scikit-learn (Random Forest), TensorFlow/Keras (CNN)

\- \*\*Backend:\*\* FastAPI

\- \*\*Frontend:\*\* HTML, CSS, JavaScript (vanilla)



\## How It Works



1\. \*\*Data pipeline\*\* — 25,000 labeled cat/dog images are read, resized to 64x64, and normalized.

2\. \*\*Baseline model\*\* — A Random Forest classifier is trained on flattened pixel arrays as a starting benchmark.

3\. \*\*CNN model\*\* — A 3-layer convolutional neural network learns spatial features (edges, textures, shapes) instead of raw pixels, improving accuracy by \~19 percentage points.

4\. \*\*API\*\* — FastAPI serves the trained CNN through a `/predict` endpoint that accepts an image upload and returns a label with confidence score.

5\. \*\*Frontend\*\* — A simple drag-and-drop interface lets users upload a photo and see the prediction in real time.





\## Project Structure



```

image-classifier-app/

├── train.py            # Random Forest training pipeline

├── train\_cnn.py        # CNN training pipeline

├── app.py              # FastAPI backend

├── index.html          # Frontend interface

├── requirements.txt    # Python dependencies

└── dataset/            # Training images (not included, see below)

```

\## Running Locally



1\. Clone the repo and install dependencies:

```bash

git clone https://github.com/shido4869/image-classifier-app.git

cd image-classifier-app

pip install -r requirements.txt

```



2\. Download the dataset from \[Kaggle: Cat and Dog Images](https://www.kaggle.com/datasets/tongpython/cat-and-dog) and place it in a `dataset/` folder.



3\. Train the model:

```bash

python train\_cnn.py

```



4\. Start the API server:

```bash

uvicorn app:app --reload

```



5\. Open `index.html` in your browser to use the app.



\## Key Learnings



\- Traditional ML models like Random Forest struggle with raw pixel data because they treat each pixel independently, missing spatial relationships.

\- CNNs use convolutional layers to learn hierarchical features — edges in early layers, shapes and textures in deeper layers — which is why they outperform classical models on image tasks.

\- Building the full pipeline (data → model → API → UI) highlighted the importance of consistent preprocessing between training and inference.



\## Future Improvements



\- Deploy the API publicly (Render/Railway)

\- Expand to multi-class classification

\- Add data augmentation to reduce overfitting and improve CNN generalization

