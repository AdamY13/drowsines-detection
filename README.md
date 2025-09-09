# 🚗 Drowsiness Detection System

A computer vision project that detects driver drowsiness in real time using eye state detection (open/closed eyes). The system raises an alert if the driver is likely falling asleep, helping to prevent accidents.

### 📌 Features

- Detects whether eyes are open or closed using a trained ML/DL model.

- Real-time webcam/video input support.

- Works with Raspberry Pi or standard PCs.

- Triggers an alert when prolonged eye closure (drowsiness) is detected.

- Lightweight and efficient for real-world deployment.

### 🧰 Tech Stack

- Programming Language: Python

- Libraries & Tools:

- OpenCV (image processing, video capture)

  - NumPy, Pandas (data handling)

  - TensorFlow / Keras (deep learning model training)

  - Scikit-learn (ML experiments)

  - Hardware (optional): Raspberry Pi + USB Camera

### 📂 Dataset

- Dataset used: Open/Closed Eyes Dataset (Kaggle)

- Contains labeled images of open and closed eyes.

- Preprocessing steps:

  - Resizing images

  - Normalization

  - Data augmentation (rotation, flipping, brightness adjustment)

### ⚙️ How It Works

- Eye images are fed into the trained CNN/ML model.

- The model classifies the state as Open or Closed.

- If eyes remain closed for a threshold duration → system detects drowsiness.

- An alert (sound/message) is triggered.
