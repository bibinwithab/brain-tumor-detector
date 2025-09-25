# Brain Tumor Segmentation using U-Net and Transfer Learning

This project implements and compares two deep learning models for the semantic segmentation of brain tumors from MRI scans. The primary goal is to accurately delineate tumor regions, providing a valuable tool for medical diagnostics.

---

The project evaluates two architectures:
- A Standard U-Net built from scratch.
- A U-Net with a VGG16 Encoder, leveraging transfer learning and a two-stage fine-tuning process for enhanced accuracy.

---

The entire pipeline, from data preparation to model deployment via a REST API, is included.
Features
- Model Comparison: Trains and evaluates two U-Net architectures to empirically determine the most effective approach.
- Transfer Learning & Fine-Tuning: Implements a robust two-stage training strategy to adapt a pre-trained VGG16 model to the medical imaging domain.
- Comprehensive Evaluation: Generates a detailed performance report (.csv) and visual graphs (training history, score box plots, prediction examples).
- REST API: A production-ready backend built with FastAPI and Uvicorn to serve the trained models and provide predictions on new images.
- Modular Codebase: The code is structured into reusable modules for configuration, model loading, utilities, and application logic.
- Standalone Inference Script: Includes a test.py script for quick, command-line-based prediction on a single image.

---

## Project Structure

```
brain-tumor-segmentation/
│
├── models/
│   ├── standard_unet_model.h5
│   └── transfer_unet_model_finetuned.h5
│
├── results/
│   ├── comparison_report.csv
│   └── (Generated graphs and plots...)
│
├── templates/
│   └── index.html
│
├── brain-tumor-dataset-segmentation-and-classification/
│   └── (Your dataset folder)
│
├── app.py                  # Main FastAPI application
├── config.py               # Configuration settings
├── model_loader.py         # Loads Keras models and custom objects
├── run_comparison_with_finetuning.py  # Main script for training and evaluation
├── test.py                 # Standalone script for testing a single image
├── utils.py                # Core processing and utility functions
└── requirements.txt        # Project dependencies
```

---

## Setup and Installation

1. Clone the Repository
```bash
git clone <your-repository-url>
cd brain-tumor-segmentation
```

2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install Dependencies
```bash
Install all the required libraries using the requirements.txt file.
pip install -r requirements.txt
```

4. Download the Dataset
Download the "Brain Tumor Segmentation and Classification" dataset and place it in the root of the project directory. The folder should be named brain-tumor-dataset-segmentation-and-classification.
---

## Usage Guide

There are three main ways to interact with this project:

1. Run the Full Training and Evaluation Experiment

This script is the core of the project. It trains both models, evaluates them, and saves the models and result graphs.
```bash
python comparison.py
```

After running, check the `models/` folder for the saved .h5 files and the `results/` folder for the performance analysis.
2. Test a Single Image

The test.py script allows you to quickly get a visual prediction for a single MRI scan using the pre-trained models.
*Ensure the models exist in the `./models/` folder first*
```python
python test.py --image path/to/your/mri_image.png
```

3. Run the API for Deployment
The FastAPI application serves the trained models via a REST API.

Step 1: Start the Server
Make sure your trained models are in the `models/` folder. Run the Uvicorn server from the project's root directory:
```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

Step 2: Test with Postman

`Method`: `POST`  

`URL`: `http://localhost:8000/predict`

`Body`: `form-data`

`KEY`: `file (Change the type from "Text" to "File")`

`VALUE`: `Select your MRI image file`

The API will return a JSON response with the base64-encoded segmentation masks if a tumor is detected.

---

## Results and Conclusion

The evaluation script (comparison.py) generates a comprehensive analysis in the `results/` folder.

Based on our experiments, the U-Net with a fine-tuned VGG16 encoder consistently outperforms the standard U-Net. The two-stage training process successfully adapts the pre-trained features to the medical domain, resulting in higher Dice and IoU scores. This demonstrates the effectiveness of transfer learning for medical image segmentation tasks.

The final report and graphs provide quantitative evidence supporting this conclusion.
