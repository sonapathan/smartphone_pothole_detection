🚗 Smartphone Detection System Using YOLOv8 for Real-Time Day and Night Road Condition Monitoring

📌 Project Overview

This project presents a real-time pothole and sinkhole detection system designed for smartphones, leveraging the YOLOv8 deep learning model. The system is capable of detecting potholes in day and night environments with support from smartphone sensors (GPS, accelerometer). It enhances road safety monitoring by providing real-time alerts and location tagging of hazards.

⚙️ Features

📱 Smartphone-Compatible: Works on Android via Kivy & Buildozer

🌗 Day & Night Detection: Integrated opencv for low-light enhancement

🎙 Voice Control: Hands-free operation using offline speech recognition (Vosk)

📍 GPS Integration: Location tagging for pothole reporting

📊 Real-Time Alerts: Hazard detection notifications for drivers

⚡ Optimized Models: YOLOv8 with pruning & quantization for mobile deployment

🛠️ Tech Stack

Programming Language: Python

Frameworks: Kivy (GUI), Flask (optional backend)

Deep Learning: YOLOv8 (Ultralytics)

Image Enhancement: opencv (for low-light images)

Voice Recognition: Vosk (offline speech-to-text)

Deployment: Buildozer (Android APK)

📥 Installation & Setup
🔹 Clone Repository
git clone https://github.com/sonapathan/smartphone_pothole_detection.git
cd smartphone_pothole_detection
📁 Roboflow Models / Data Sources
Daytime Model: Roboflow Day Model
Nighttime Model: Roboflow Night Model
These links point to the datasets/models used for training the YOLOv8 detection system.

🔹 Install Dependencies
pip install -r requirements.txt

🔹 Run Application (Desktop Test)
python main.py

🔹 Build for Android
buildozer init
buildozer -v android debug

🚀 Usage Instructions

Open the app on your smartphone.

Allow camera & GPS permissions.

Real-time detection begins automatically.

Use voice commands to start/stop detection (Vosk).

Alerts + GPS coordinates are logged for reporting potholes.

📊 Results

The system was tested under daytime and nighttime conditions.

Metric	Daytime	Nighttime	Description
Precision	0.877	0.8794	Accuracy of positive pothole detections
Recall	0.848	0.7907	Proportion of actual potholes correctly detected
mAP@0.5	0.863	0.8639	Mean Average Precision at IoU threshold 0.5
mAP@0.5:0.95	0.576	0.5093	Mean Average Precision across stricter IoU thresholds
F1-Score	0.862	0.833	Harmonic mean of Precision and Recall
Box Loss	1.218	1.2182	Localization error of bounding boxes
Class Loss	0.670	0.6696	Classification error of detected objects
DFL Loss	1.237	1.2373	Distribution Focal Loss for bounding box regression
Confusion Matrix	✔️	✔️	Visualizes true positives, false positives, and false negatives
🔹 Confusion Matrices

Daytime Evaluation
(Insert daytime confusion matrix image here)

Nighttime Evaluation
(Insert nighttime confusion matrix image here)

👨‍🏫 Contributors

Mr. Zeeshan Qureshi – Lecturer, Department of Information and Computing , University of Sufism and Modern Sciences, Bhit Shah

Sona Pathan – Research Scholar, University of Sufism and Modern Sciences, Bhit Shah

Mehwish Rajput – Research Scholar, University of Sufism and Modern Sciences, Bhit Shah

Tamsila Mirjat – Research Schola, University of Sufism and Modern Sciences, Bhit Shah
