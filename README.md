🌿 Plant Doctor AI

AI-powered Plant Disease Detection & Treatment Advisor

A full-stack deep learning web application that diagnoses plant diseases from leaf images and provides scientifically grounded treatment recommendations using a Retrieval-Augmented Generation (RAG) system.

Live Demo: https://plant-doctor-deploy.onrender.com/dashboard
Repository: https://github.com/Conspirer/imd-project


🚀 What This Project Does

Plant Doctor AI allows users to:

Upload a photo of a plant leaf

Automatically detect the disease using a deep neural network

Visualize where the AI is focusing using Grad-CAM heatmaps

Get actionable treatment recommendations from a medical-style knowledge base

Ask follow-up questions using an AI-powered chatbot

It combines computer vision, deep learning, explainable AI, and retrieval-augmented generation into a single deployed system.


🧠 AI Architecture
1. Disease Classification Model

The core model is a MobileNet-V2 convolutional neural network trained on the PlantVillage dataset.

Input: RGB leaf image (224×224)

Backbone: Pretrained MobileNet-V2 (ImageNet)

Output: Softmax probabilities over plant disease classes

Loss: Cross-Entropy

Optimizer: Adam

Training: Transfer learning + data augmentation

2. Explainable AI (Grad-CAM)

After prediction, the system runs Gradient-weighted Class Activation Mapping (Grad-CAM) to show:

Which regions of the leaf influenced the prediction most

This builds trust in the AI by visually highlighting disease-affected areas.

📚 RAG (Retrieval-Augmented Generation)

The treatment engine is powered by a custom RAG pipeline:

Knowledge base: rag_kb/treatments.md

Indexed by disease name, symptoms, and treatments

Uses semantic similarity to retrieve the most relevant medical guidance

The prediction class is used as the primary query

This ensures the AI never hallucinates treatments — it always retrieves real curated content.


🖥 Web Application Stack
Layer	Technology
Frontend	HTML, CSS, JavaScript
Backend	FastAPI
AI Framework	PyTorch
Image Processing	OpenCV
Model Serving	Uvicorn
Deployment	Render
Dataset	PlantVillage


📂 Project Structure
plant-doctor-ai/
├── app.py              # FastAPI server
├── train.py            # Model training pipeline
├── infer.py            # CLI inference with Grad-CAM
├── grad_cam.py         # Explainable AI
├── rag.py              # RAG engine
├── utils.py            # Image preprocessing
├── models/
│   └── plant_doctor.pt # Trained neural network
├── rag_kb/
│   └── treatments.md  # Medical knowledge base
├── static/             # Frontend UI
└── requirements.txt


🧪 How It Works (Flow)

User uploads leaf image

Image → CNN → Disease probabilities

Grad-CAM generates heatmap

RAG retrieves treatment text

UI displays:

Disease name

Confidence

Heatmap

Treatment plan



🏗 Model Training

The model is trained using:

Transfer learning from ImageNet

Data augmentation (rotation, flipping)

Validation split to avoid overfitting

Best-model checkpointing

Saved as:

models/plant_doctor.pt


🌍 Deployment

The app is deployed on Render as a cloud-hosted FastAPI service.

The model file is loaded dynamically and runs entirely on CPU — no GPU required.
