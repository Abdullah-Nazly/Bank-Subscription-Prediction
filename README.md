# Bank Marketing Campaign - Neural Network Classification

A machine learning project that predicts whether a bank customer will subscribe to a term deposit using a neural network classifier built with TensorFlow/Keras.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Experiments & Results](#experiments--results)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project implements a binary classification model to predict bank term deposit subscriptions. The model uses a deep neural network to analyze customer characteristics and campaign information to determine the likelihood of a customer subscribing to a term deposit.

**Target Variable**: `y` (yes/no) - Whether the customer subscribed to a term deposit

## 📊 Dataset

- **Source**: Bank Marketing Dataset (`bank-full.csv`)
- **Size**: 45,211 records (45,210 after outlier removal)
- **Features**: 17 original features
- **Delimiter**: Semicolon (`;`)

### Dataset Characteristics
- **No missing values**: All features are complete
- **No duplicates**: Clean dataset
- **Class Imbalance**: Significant imbalance between classes (majority: 'no', minority: 'yes')

## 🔍 Features

### Numerical Features (7)
- `age`: Customer age
- `balance`: Account balance
- `day`: Day of month (removed during preprocessing)
- `duration`: Last contact duration (seconds)
- `campaign`: Number of contacts during this campaign
- `pdays`: Days since last contact (categorized)
- `previous`: Number of contacts before this campaign

### Categorical Features (10)
- `job`: Type of job (one-hot encoded)
- `marital`: Marital status (one-hot encoded)
- `education`: Education level (label encoded)
- `default`: Has credit in default (label encoded: yes=1, no=0)
- `housing`: Has housing loan (label encoded: yes=1, no=0)
- `loan`: Has personal loan (label encoded: yes=1, no=0)
- `contact`: Contact communication type (one-hot encoded)
- `month`: Month of last contact (removed during preprocessing)
- `poutcome`: Outcome of previous campaign (label encoded)
- `y`: Target variable (label encoded: yes=1, no=0)

## 🔧 Data Preprocessing

### 1. Data Quality Checks
- ✅ Missing values: None found
- ✅ Duplicate rows: None found
- ✅ Data types: Verified and converted as needed

### 2. Feature Engineering

#### Statistical Analysis
- **Chi-square tests**: Used to determine feature importance for categorical variables
  - `contact`: p-value < 0.05 (highly significant)
  - `poutcome`: p-value = 0.0 (extremely significant)
  - `marital`: p-value < 0.05 (significant)

- **Point-Biserial Correlation**: Used for numerical features
  - `duration`: Highest correlation (0.395)
  - `pdays`: Moderate correlation (0.104)
  - `previous`: Moderate correlation (0.093)
  - `day`: Low correlation (-0.028) - **Removed**

#### Encoding Strategies
- **One-Hot Encoding**: Applied to `contact`, `job`, `marital`, and `pdays` (categorized)
- **Label Encoding**: Applied to `education`, `poutcome`, `default`, `housing`, `loan`
- **Binary Encoding**: Applied to yes/no features

#### Feature Removal
- `day`: Low correlation with target variable
- `month`: Similar characteristics to `day`
- One outlier in `previous` column (value > 250)

#### Feature Transformation
- **pdays Categorization**: Converted to categories:
  - Not Contacted (-2 to 0)
  - Recently Contacted (0 to 100)
  - Contacted Long Ago (100 to 300)
  - Very Long Ago (300 to 900)

### 3. Data Scaling
- **StandardScaler**: Applied to all features for neural network training
- Normalizes features to have mean=0 and std=1

## 🏗️ Model Architecture

### Base Model
```python
Sequential([
    Dense(10, activation='relu', input_shape=(33,)),
    Dense(1, activation='sigmoid')
])
```

### Optimized Model (Best Performance)
```python
Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])
```

### Hyperparameters
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Regularization**: L2 regularization (λ=0.01)
- **Batch Normalization**: Applied after each hidden layer

## 🧪 Experiments & Results

### Experiment 1: Raw vs Standardized Data
- **Raw Data Model**: 89.22% accuracy
- **Standardized Data Model**: 89.76% accuracy
- **Conclusion**: Standardization improves model performance

### Experiment 2: Class Weights
- **Class Weights**: {0: 0.566, 1: 4.310}
- **Result**: Improved recall for minority class
- **Trade-off**: Slight decrease in overall accuracy but better balanced predictions

### Experiment 3: Hyperparameter Tuning
- **Method**: Keras Tuner (Hyperband)
- **Tuned Parameters**:
  - Number of layers: 1-3
  - Neurons per layer: 8-64
  - Learning rate: 1e-5 to 1e-2
- **Best Configuration**: 
  - Units1: 8
  - Additional layers with 24, 48, 40 neurons
  - Learning rate: 0.0086

### Experiment 4: SMOTE + Regularization (Best Model)
- **SMOTE**: Synthetic Minority Oversampling Technique
- **Architecture**: 2 hidden layers (64 neurons each)
- **Regularization**: L2 + Batch Normalization
- **Results**:
  - **Accuracy**: ~88-89%
  - **ROC-AUC**: Calculated and visualized
  - **Threshold Tuning**: Tested thresholds (0.71, 0.73, 0.75, 0.78)

### Final Model Performance
- **Test Accuracy**: ~88-89%
- **Precision (Class 0)**: ~94%
- **Precision (Class 1)**: ~52-56%
- **Recall (Class 0)**: ~93-95%
- **Recall (Class 1)**: ~49-57%

## 💻 Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras-tuner imbalanced-learn scipy
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras-tuner
imbalanced-learn
scipy
```

## 🚀 Usage

1. **Prepare the Dataset**
   - Place `bank-full.csv` in the `./Dataset/` directory
   - Ensure the file uses semicolon (`;`) as delimiter

2. **Run the Notebook**
   - Open `Neural_Network.ipynb` in Jupyter Notebook
   - Execute cells sequentially

3. **Model Training**
   - The notebook includes multiple model experiments
   - Final model uses SMOTE and regularization
   - Training includes validation split (20%)

4. **Evaluation**
   - Classification reports
   - Confusion matrices
   - ROC curves
   - Threshold optimization

## 📈 Key Findings

1. **Feature Importance**:
   - `duration`: Most important numerical feature (correlation: 0.395)
   - `contact`, `poutcome`, `marital`: Most important categorical features

2. **Data Preprocessing Impact**:
   - Standardization significantly improves model performance
   - Feature selection (removing `day` and `month`) reduces noise

3. **Class Imbalance Handling**:
   - SMOTE provides better balance than class weights alone
   - Threshold tuning helps optimize precision-recall trade-off

4. **Model Architecture**:
   - Deeper networks (2-3 layers) with regularization perform best
   - Batch normalization stabilizes training
   - L2 regularization prevents overfitting

5. **Performance Metrics**:
   - High accuracy on majority class (Class 0)
   - Moderate performance on minority class (Class 1)
   - ROC-AUC provides comprehensive model evaluation

## 🛠️ Technologies Used

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, tensorflow, keras
- **Hyperparameter Tuning**: keras-tuner
- **Imbalanced Learning**: imbalanced-learn (SMOTE)
- **Statistical Analysis**: scipy

## 📝 Notes

- The dataset is imbalanced, which affects model performance on the minority class
- Multiple experiments were conducted to find the optimal model configuration
- Feature engineering played a crucial role in model performance
- Regularization and batch normalization help prevent overfitting

## 🔮 Future Improvements

- Experiment with more complex architectures (e.g., deeper networks)
- Try different oversampling techniques (ADASYN, BorderlineSMOTE)
- Implement ensemble methods
- Feature importance analysis using SHAP values
- Cross-validation for more robust evaluation

---

**Project Type**: Binary Classification  
**Domain**: Banking/Finance  
**Model Type**: Deep Neural Network  
**Framework**: TensorFlow/Keras
