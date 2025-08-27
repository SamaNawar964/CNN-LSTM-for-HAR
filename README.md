# 🏃 Human Activity Recognition using CNN-LSTM

This project implements a **CNN-LSTM hybrid model** for recognizing human activities from time-series sensor data.  
The model is trained on CSV files (`train.csv` and `test.csv`) where each row represents extracted features from sensor signals and the last column is the activity label.

---

## 📂 Project Structure
```

├─ har\_cnn\_lstm.ipynb   # main notebook (data, preprocessing, training, evaluation, predictions)
├─ train.csv            # training dataset
├─ test.csv             # test dataset
└─ requirements.txt     # dependencies

````

---

## ⚙️ Setup

Clone the repository:
```bash
git clone https://github.com/SamaNawar964/cnn-lstm-har.git
cd cnn-lstm-har
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook har_cnn_lstm.ipynb
```

---

## 🧠 Model Architecture

* **Conv1D + BatchNormalization + MaxPooling + Dropout** (feature extraction)
* **Conv1D + BatchNormalization + MaxPooling + Dropout** (deeper features)
* **LSTM (64 units)** (temporal learning)
* **Dense (64, relu) + Dropout**
* **Output Layer (Softmax)**

---

## 📊 Results

* **Validation Accuracy**: \~95%
* **Test Accuracy**: \~95%

Example prediction:

```
Predicted Class: [3]  
Predicted Activity: Sitting
```

---

## 📈 Training Curves

The notebook plots:

* Accuracy (Training vs Validation)
* Loss (Training vs Validation)

---

## 🛠️ Tech Stack

* Python 3.x
* TensorFlow / Keras
* NumPy, Pandas
* Scikit-learn
* Matplotlib

---

## 🚀 Future Improvements

* Add more layers or try **GRU** instead of LSTM.
* Use raw accelerometer/gyroscope signals with sliding windows.
* Deploy as a real-time mobile app.

---

## 📌 Acknowledgements

* Inspired by research on deep learning for **Human Activity Recognition (HAR)**.
* Dataset format: CSV files with features + labels.
