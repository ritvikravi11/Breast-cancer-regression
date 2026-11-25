# Breast Cancer Detection using Regression

This project aims to develop a machine learning model for breast cancer detection using regression techniques. The model will analyze various features related to breast cancer and predict the likelihood of cancer presence.

## Project Structure

```
breast-cancer-regression-ml
├── data
│   ├── raw
│   │   └── custom_dataset.csv
│   └── processed
│       └── dataset.csv
├── notebooks
│   └── exploration.ipynb
├── src
│   ├── app.py
│   ├── data
│   │   └── prepare.py
│   ├── models
│   │   └── train.py
│   ├── inference
│   │   └── predict.py
│   └── types
│       └── index.py
├── tests
│   └── test_train.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Dataset

The dataset used in this project is located in the `data/raw/custom_dataset.csv` file. It contains features such as tumor size, age, and other relevant attributes, along with a target variable representing the likelihood of cancer.

The processed version of the dataset can be found in `data/processed/dataset.csv`, which includes cleaned data and any transformations applied to prepare the data for modeling.

## Notebooks

The `notebooks/exploration.ipynb` file is used for exploratory data analysis (EDA). It includes visualizations and statistical summaries to help understand the dataset better and identify patterns or correlations.

## Source Code

- `src/app.py`: Main entry point for the application, initializing the machine learning pipeline.
- `src/data/prepare.py`: Functions for data preparation, including loading and cleaning the dataset.
- `src/models/train.py`: Implementation of the regression model training process.
- `src/inference/predict.py`: Functions for making predictions using the trained model.
- `src/types/index.py`: Exports data types and interfaces used throughout the project.

## Testing

Unit tests for the model training functions are located in `tests/test_train.py` to ensure that the training process works as expected.

## Requirements

The required Python dependencies for this project are listed in the `requirements.txt` file.

## Setup Instructions

1. Clone the repository.
2. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```
3. Prepare the dataset by running the data preparation script.
4. Train the model using the training script.
5. Use the inference script to make predictions.

## Usage

Follow the instructions in the Jupyter notebook for exploratory data analysis and model training. The application can be run using the main entry point script.