#!/usr/bin/env python
# coding: utf-8

# #### `Note: Understanding the Problem`
# Customer churn prediction involves identifying customers likely to stop using a service. Businesses use this information to take proactive measures to retain customers. We’ll use a public dataset and build a deep learning model to predict churn.
# 
# Customer churn is the percentage of customers who stop using a business's products or services over a specific period of time

# ## `Import Libraries`

# In[1]:


# Importing Required Libraries

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sn
import pickle
import os

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Deep Learning Models
import tensorflow as tf
from tensorflow.keras.models import Sequential          # Sequential: A simple way to build a model layer by layer (like stacking building blocks).
from tensorflow.keras.layers import Dense               # Dense: A fully connected layer where each neuron is connected to every neuron in the previous and next layers.
from tensorflow.keras.layers import Dropout             # Dropout: A technique to prevent the model from overfitting by randomly “turning off” some neurons during training.


# ## `Reading Dataset`

# In[2]:


# Load the dataset
data = pd.read_csv('telco_customer_churn.csv')

# Display the first few rows
data.head()


# In[3]:


# Check for missing values
data.isnull().sum()


# In[4]:


# Data summary
data.info()


# ## `Preprocessing Dataset`

# In[5]:


# Drop irrelevant columns (The colums isn't required further in the analysis)
data = data.drop(columns=['customerID'])


# In[6]:


# Convert categorical columns to numerical
label_encoders = {} # This is a dictionary where the encoding will be stored for each column. So that if/when required we can inverse transform the data using the stored encoders. 
for column in data.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])


# In[7]:


# Encode target column
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[8]:


# Scale numerical features
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
data[numerical_features] = scaler.fit_transform(data[numerical_features])


# In[9]:


# Split into features (X) and target (y)
X = data.drop(columns=['Churn'])
y = data['Churn']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## `Deep Learning Model Build`

# `Note to Self:` What is Happening Here? - 
# You are building a neural network model for binary classification (predicting if something belongs to one of two classes, like “Yes” or “No”) using TensorFlow and Keras. 

# In[10]:


# What it does: Starts an empty model where you’ll add layers one by one.
# Think of it like this: You’re building a sandwich and starting with an empty plate.

# Initialize the model
model = Sequential()


# In[11]:


# Input layer
model.add(Dense(100,                             # Dense(64): Creates a layer with 64 neurons.
                input_dim=X_train.shape[1],     # input_dim=X_train.shape[1]: Tells the model how many input features there are (the number of columns in your dataset). Each feature will have its own “neuron.”
                activation='relu'))             # activation='relu': Applies the ReLU (Rectified Linear Unit) activation function to each neuron. What it does: Keeps only positive values (turns negatives into 0), which helps the model learn complex patterns.


# In[12]:


# Hidden layers (Hidden layer role: Think of it as the brain solving smaller problems to understand the big picture.)
model.add(Dense(64, activation='relu')) # Dense(32): Adds another layer with 32 neurons. These neurons process information from the previous layer.

# Drop Neurons (This prevents the model from memorizing the data (overfitting) and helps it generalize better to unseen data.)
model.add(Dropout(0.3))  # Dropout for regularization. Dropout(0.3): During training, 30% of neurons in this layer are randomly “turned off” for each training step.


# In[13]:


# Output layer
model.add(Dense(1,                      # # Dense(1): The output layer has 1 neuron because this is a binary classification problem (predicting “Yes” or “No”).
                activation='sigmoid'))  # activation='sigmoid': Outputs a value between 0 and 1. This represents the probability of belonging to one class (e.g., “Yes”).


# In[14]:


# Compile the model
model.compile(loss='binary_crossentropy', # Measures how far off the predictions are for binary classification problems. Lower loss = better model performance.
              optimizer='adam',           # A method to update the model’s weights to reduce the loss. Think of it as the “coach” guiding the model to improve after every mistake. 
              metrics=['accuracy'])       # Tracks how often the model predicts correctly during training.


# In[15]:


# Summary of the model
model.summary()


# In[16]:


# Train the model
history = model.fit(X_train, y_train, epochs=25,                         # epochs: The number of iterations over the entire dataset.
                                      batch_size=32,                     # batch_size: The number of samples per gradient update.
                                      validation_data=(X_test, y_test))  # validation_data: Evaluates the model on test data after each epoch.


# In[22]:


prediction = model.predict(X_test)[0][0]
prediction


# In[17]:


import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()

# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")


# In[18]:


os.chdir(r'/Users/sakshamarora/Documents/3. Work Stuff/Python_3.11/10. GitHub Project (Career Focused)/2. Customer Churn Prediction Streamlit Application')

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)


# In[19]:


# Save the trained model
model.save('customer_churn_model.h5')
# model.save('my_model.keras') : New way to save the model.

