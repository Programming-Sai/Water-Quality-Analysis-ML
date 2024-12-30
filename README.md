# Water-Quality-Analysis-ML

---

## Project Overview

This project aims to predict water quality using a dataset of various physicochemical properties. By leveraging machine learning models, we aim to classify water samples as safe or unsafe based on their attributes.

## Dataset

- **Source:** [Kaggle - Water Quality Dataset](https://www.kaggle.com/datasets/ozgurdogan646/water-quality-dataset)
- **Description:** The dataset contains measurements such as pH, turbidity, conductivity, and other parameters used to assess water quality.

## Goals

1. Explore the dataset and perform data visualization.
2. Preprocess the data by handling missing values and scaling features.
3. Train machine learning models to predict water quality.
4. Evaluate and compare model performance.
5. Deploy the final model using a Flask-based web app.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Programming-Sai/Water-Quality-Analysis-ML.git
   cd Water-Quality-Analysis-ML
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Folder Structure

```
water-quality-prediction/
│
├── data/
│   └── water_quality.csv        # The dataset
│
├── notebooks/
│   └── data_exploration.ipynb   # For data exploration and visualization
│   └── model_building.ipynb     # For training the model
│
├── src/
│   ├── train.py                 # Training script
│   ├── predict.py               # Prediction script
│   └── utils.py                 # Utility functions
│
├── models/
│   └── water_quality_model.pkl  # Saved model file
│
├── app/
│   ├── app.py                   # Flask app for deployment
│   └── templates/               # (Optional) HTML templates for the app
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

## Usage

1. Explore the dataset:
   - Open and run `notebooks/data_exploration.ipynb` to understand the data and visualize distributions.
2. Train the model:
   - Use `notebooks/model_building.ipynb` or `src/train.py` to train and evaluate the machine learning models.
3. Predict water quality:
   - Use `src/predict.py` to make predictions on new data.
4. Deploy the model:
   - Run the Flask app using:
     ```bash
     python app/app.py
     ```

## Key Libraries

- **pandas**: Data manipulation
- **seaborn & matplotlib**: Data visualization
- **scikit-learn**: Machine learning models
- **Flask**: Model deployment

## Contributors

- [Mensah Isaiah](https://github.com/Programming-Sai)
- [Contributor 2](https://github.com/contributor-username)
- [Contributor 3](https://github.com/contributor-username)

---

### **Steps to Set It Up**

1. Create a new repository on GitHub with the name `water-quality-prediction`.
2. Initialize your project folder locally and link it to the GitHub repo:
   ```bash
   git init
   git remote add origin https://github.com/Programming-Sai/Water-Quality-Analysis-ML.git
   git branch -M main
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

---
