
# Customer Churn Prediction

This project uses a deep learning model to predict customer churn. Customer churn is the percentage of customers who stop using a service over a specific time. The model uses a public dataset and a neural network built with TensorFlow/Keras.

## Features
- Data preprocessing: Handles missing values, encodes categorical features, and scales numerical features.
- Deep Learning Model: A neural network with input, hidden, and output layers for binary classification.
- Model Training: Includes dropout layers for regularization and validation on test data.
- Outputs: Churn probability for each customer and accuracy metrics.

## Requirements
- Python 3.6 or higher
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tensorflow`
  - `pickle`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
- Dataset: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Data includes customer demographics, account information, and churn status.

## Project Structure
- `customer_churn_model.h5`: Saved trained model.
- `scaler.pkl` and `encoders.pkl`: Saved preprocessing objects.
- `churn.ipynb`: Jupyter Notebook with the complete workflow.


## Results
- **Model Accuracy**: ~85% (depending on training parameters)
- Visualizations include training/validation accuracy per epoch.

## Future Enhancements
- Add feature importance analysis.
- Integrate with a Streamlit app for real-time predictions.
- Experiment with different deep learning architectures.

