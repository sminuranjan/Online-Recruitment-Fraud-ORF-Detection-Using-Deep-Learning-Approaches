# Online-Recruitment-Fraud-ORF-Detection-Using-Deep-Learning-Approaches


## Project Description

Online Recruitment Fraud (ORF) Detection Using Deep Learning Approaches** is a machine learning-based system that detects fraudulent job postings. Using Natural Language Processing (NLP) and Deep Learning algorithms, it analyzes job descriptions and related information to classify postings as genuine or fake. The system helps job seekers avoid scams and enhances recruitment security.


---

## Objectives

* Detect fraudulent job advertisements automatically.
* Reduce the risk of online recruitment scams.
* Improve job seeker safety.
* Apply Deep Learning techniques for text classification.
* Enhance recruitment platform security.

---

## Features

* Data preprocessing and cleaning.
* Text feature extraction using NLP techniques.
* Deep Learning-based classification model.
* Fraudulent and genuine job prediction.
* Model performance evaluation.
* User-friendly prediction interface.

---

## Technologies Used

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* Flask

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn

### Database

* MySQL (Optional)

---

## System Requirements

### Hardware Requirements

* Processor: Intel Core i3 or above
* RAM: 4 GB minimum
* Storage: 10 GB free space

### Software Requirements

* Python 3.8+
* Jupyter Notebook
* VS Code / PyCharm
* Flask
* TensorFlow
* Keras

---

## Dataset

The project uses the Online Recruitment Fraud Detection dataset containing various job postings with features such as:

* Job Title
* Company Profile
* Description
* Requirements
* Benefits
* Employment Type
* Industry
* Fraudulent Label

Dataset Source:

* Kaggle ORF Dataset

---

## 📂 Project Structure

```text
Online-Recruitment-Fraud-Detection/
│
├── 📁 dataset/
│   └── fake_job_postings.csv          # Recruitment fraud dataset
│
├── 📁 models/
│   └── fraud_detection_model.h5       # Trained deep learning model
│
├── 📁 notebooks/
│   └── training.ipynb                 # Model training and experimentation
│
├── 📁 static/
│   ├── 📁 css/
│   │   └── style.css                  # Stylesheets
│   │
│   ├── 📁 images/
│   │   └── logo.png                   # Project images
│
├── 📁 templates/
│   ├── index.html                     # Home page
│   └── result.html                    # Prediction result page
│
├── app.py                             # Flask web application
├── train_model.py                     # Model training script
├── requirements.txt                   # Project dependencies
├── README.md                          # Project documentation
└── LICENSE                            # License information
```

---

## Working Process

### Step 1: Data Collection

The dataset containing genuine and fraudulent job postings is collected.

### Step 2: Data Preprocessing

* Remove missing values.
* Remove special characters.
* Convert text to lowercase.
* Tokenization and text cleaning.

### Step 3: Feature Extraction

Text data is converted into numerical form using:

* TF-IDF Vectorization
* Word Embeddings

### Step 4: Model Training

A Deep Learning model such as:

* Artificial Neural Network (ANN)
* Long Short-Term Memory (LSTM)

is trained on the processed dataset.

### Step 5: Prediction

The trained model predicts whether a job posting is:

* Genuine (0)
* Fraudulent (1)

### Step 6: Result Display

The prediction result is displayed through a web interface.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Online-Recruitment-Fraud-Detection.git
```

### Navigate to Project Directory

```bash
cd Online-Recruitment-Fraud-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Train the Model

```bash
python train_model.py
```

### Run the Flask Application

```bash
python app.py
```

### Open Browser

```text
http://127.0.0.1:5000
```

---

## Output

### Input

Job Description:

```
Work from home and earn $5000 weekly with no experience required.
```

### Prediction

```
Fraudulent Job Posting
```

---

## Future Enhancements

* Integration with real-time job portals.
* Advanced NLP models such as BERT.
* Mobile application support.
* Real-time fraud monitoring dashboard.
* Multi-language job fraud detection.

---

## Conclusion

The Online Recruitment Fraud Detection System helps identify fake job advertisements using Deep Learning and NLP techniques. The project contributes to creating a safer online recruitment environment by automatically detecting suspicious job postings and protecting job seekers from fraud.

---

### requirements.txt

```txt
numpy
pandas
matplotlib
scikit-learn
tensorflow
keras
flask
nltk
```

