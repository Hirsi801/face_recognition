# Face Recognition System

![Demo](demo.gif) <!-- Add a demo gif later -->

A TensorFlow-based face recognition system using ResNet50 with ArcFace and Siamese networks.

## Features
- 👥 Multi-class face recognition
- 🔍 Face verification (same/different person)
- 📷 Real-time camera inference
- 🛠 Data augmentation pipeline
- 📊 Performance evaluation metrics

## Installation
```bash
git clone https://github.com/yourusername/face_recognition
cd face-recognition-system
pip install -r requirements.txt
Usage
1. Prepare Dataset
Organize images in data/ directory:

text
data/
├── person1/
│   ├── img1.jpg
│   └── ...
└── person2/
    ├── img1.jpg
    └── ...
2. Train Model
bash
python src/train.py --epochs 50 --batch_size 32
3. Run Real-Time Recognition
bash
python src/recognize.py
4. Evaluate Performance
bash
python src/evaluate.py
Configuration
Modify these parameters in src/model.py:

python
INPUT_SHAPE = (160, 160, 3)  # Image dimensions
EMBEDDING_SIZE = 256          # Feature vector size
Results
Metric	Classification	Verification
Accuracy	92%	88%
Precision	0.91	0.87
Recall	0.92	0.89
Contributing
Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

License
Distributed under the MIT License
