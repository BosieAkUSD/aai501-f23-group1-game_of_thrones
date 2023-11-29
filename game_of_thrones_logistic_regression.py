# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def logistic_regression_analysis():
   # Step 1: Load Datasets
   battles = pd.read_csv("data_files/battles.csv")
   character_deaths = pd.read_csv("data_files/character-deaths.csv")
   character_predictions = pd.read_csv("data_files/character-predictions.csv")

   # Step 2: Explore and Preprocess Data (Already Completed)

   # Step 3: Feature Selection
   # Identify relevant features for our logistic regression model
   selected_features = ['Gender', 'Nobility', 'GoT', 'CoK', 'SoS', 'FfC', 'DwD', 'Death Year', 'Book of Death', 'Death Chapter']

   # Step 4: Define Target Variable
   target_variable = 'Death Year'  # Assuming 'Death Year' is our target variable

   # Step 5: Split Data into Training and Testing Sets
   # Combine relevant features and target variable
   data = character_deaths[selected_features + [target_variable]].copy()

   # Handle missing values
   data.fillna(0, inplace=True)  # Assuming missing values mean the absence of death

   # Split data into features (X) and target variable (y)
   X = data[selected_features]
   y = data[target_variable]

   ''' 
   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Step 6: Build Logistic Regression Model
   # Use one-hot encoding for categorical variables
   categorical_features = ['Gender', 'Nobility']
   numeric_features = ['GoT', 'CoK', 'SoS', 'FfC', 'DwD', 'Death Year', 'Book of Death', 'Death Chapter']

   # Create a column transformer
   preprocessor = ColumnTransformer(
      transformers=[
         ('num', 'passthrough', numeric_features),
         ('cat', OneHotEncoder(drop='first'), categorical_features)  # Use drop='first' to avoid multicollinearity
      ])

   # Create a logistic regression model
   model = Pipeline([
      ('preprocessor', preprocessor),
      ('classifier', LogisticRegression(random_state=42))
   ])

   # Step 7: Evaluate Model
   # Train the model
   model.fit(X_train, y_train)

   # Make predictions on the test set
   y_pred = model.predict(X_test)

   # Evaluate the model
   accuracy = accuracy_score(y_test, y_pred)
   precision = precision_score(y_test, y_pred)
   recall = recall_score(y_test, y_pred)
   f1 = f1_score(y_test, y_pred)
   conf_matrix = confusion_matrix(y_test, y_pred)

   print(f"Accuracy: {accuracy:.2f}")
   print(f"Precision: {precision:.2f}")
   print(f"Recall: {recall:.2f}")
   print(f"F1 Score: {f1:.2f}")
   print("Confusion Matrix:")
   print(conf_matrix)

   # Step 8: Interpret Coefficients 
   # Coefficients interpretation depends on the features selected and their transformations

   # Step 9: Fine-tune Model
   # Adjust model parameters or select different features based on the initial performance

   # Step 10: Save Code in a Python Script

   # Step 11: Run Script and Save Output 
''' 