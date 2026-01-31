🏜️ Offroad Semantic Scene Segmentation

Duality AI Hackathon – Segmentation Track

Team Name:3diots
📌 Overview

This project implements a semantic scene segmentation model for offroad desert environments using DeepLabV3+ with a ResNet101 backbone.
The goal is to accurately classify every pixel in an image into terrain and object classes critical for offroad autonomy.

The model is trained exclusively on the provided synthetic dataset from Duality AI and evaluated on unseen test images, following all hackathon rules.

🧠 Model Architecture

Model: DeepLabV3+

Backbone: ResNet101 (ImageNet pretrained)

Framework: PyTorch

Library: segmentation-models-pytorch

Input Resolution: 256 × 256

Loss Function: Cross Entropy Loss

Optimizer: AdamW

🗂️ Dataset Structure

The dataset provided by Duality AI is expected in the following format:

Offroad_Segmentation_Training_Dataset/
├── train/
│ ├── Color_Images/
│ └── Segmentation/
├── val/
│ ├── Color_Images/
│ └── Segmentation/
└── testImages/
└── Color_Images/

⚠️ Important:

Test images are never used during training

Training, validation, and testing remain strictly separated

🏷️ Class Mapping

Each pixel label in the segmentation masks is mapped as follows:

Class ID	Class Name
0	Background
100	Trees
200	Lush Bushes
300	Dry Grass
500	Dry Bushes
550	Ground Clutter
700	Logs
800	Rocks
7100	Landscape
10000	Sky
⚙️ Environment Setup
1️⃣ Create and activate Conda environment
conda create -n EDU python=3.10 -y
conda activate EDU
2️⃣ Install dependencies
pip install -r requirements.txt
🚀 Training the Model

Run the training script from the project root:

python train.py
Training details:

Image size: 256×256

Epochs: 8

Batch size:

CPU: 2

GPU: 4

BatchNorm layers are frozen to ensure stability with small batch sizes

The trained model is saved as:

deeplabv3plus_resnet101.pth
🧪 Testing & Evaluation

After training completes, evaluate on unseen test images:

python test.py
Evaluation Metrics:

Mean Intersection over Union (IoU)

Pixel Accuracy

Qualitative prediction visualizations

Outputs include:

Segmentation predictions

IoU score

Performance logs

📊 Results Summary

(Update this section with your final results)

Validation IoU: XX.XX

Pixel Accuracy: XX.XX

Observations:

Strong performance on ground and vegetation classes

Some confusion between visually similar classes (e.g., dry grass vs bushes)

⚠️ Known Challenges

CPU-only training significantly increases training time

Class imbalance affects rare object categories

Small batch sizes required BatchNorm freezing

🔮 Future Improvements

Train on GPU for faster convergence

Apply class-weighted loss

Use advanced data augmentation

Experiment with transformer-based backbones

📁 Repository Contents
.
├── train.py
├── test.py
├── requirements.txt
├── deeplabv3plus_resnet101.pth
├── README.md
└── Offroad_Segmentation_Training_Dataset/
📜 Hackathon Compliance

✅ Model trained only on provided dataset
✅ No test data leakage
✅ Fully reproducible setup
✅ Clear documentation provided

👥 Team Members

P YASHITHA SAI
R HARINI SRI
B VYBHAV

🔗 Useful Links

Duality AI Falcon Platform: https://falcon.duality.ai

Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch

🏁 Conclusion

This project demonstrates an effective and reproducible approach to semantic segmentation for offroad autonomy using synthetic data. The trained model achieves competitive IoU scores while maintaining compliance with all hackathon guidelines.
