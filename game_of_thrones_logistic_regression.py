import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def logistic_regression_analysis():
    print(" ")
    print("Running Logistic Regression Analysis...")
    # Load datasets
    character_predictions = pd.read_csv("data_files/character-predictions.csv")
    character_deaths = pd.read_csv("data_files/character-deaths.csv")

    # Feature Selection
    selected_features = ['isNoble', 'male', 'age', 'numDeadRelations', 'boolDeadRelations', 'isPopular', 'popularity']

    # Combine datasets based on common keys
    merged_data = pd.merge(character_predictions, character_deaths, how='left', left_on='name', right_on='Name')

    # Identify features and target variable
    X = merged_data[selected_features]
    y = merged_data['actual']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Create and train the logistic regression model
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train_imputed, y_train)

    # Make predictions on the testing set
    y_pred = logistic_regression_model.predict(X_test_imputed)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Display evaluation metrics
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", confusion_mat)

    # Step 8: Interpret Coefficients
    # Coefficients can be accessed using logistic_regression_model.coef_
    coefficients = logistic_regression_model.coef_
    print("\nCoefficients:")
    print(coefficients)

 
    feature_names = X.columns
    coefficients_with_features = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients.flatten()})
    print("\nCoefficients with Features:")
    print(coefficients_with_features)

    # Step 9: Fine-tune Model (if necessary)
    # Fine-tuning involves adjusting model parameters or selecting different features based on initial performance.
    # Reevaluate the model's performance and iterate if necessary.

    # Example: Change regularization parameter C (inverse of regularization strength)
    new_logistic_regression_model = LogisticRegression(C=0.1)
    new_logistic_regression_model.fit(X_train_imputed, y_train)

    # Evaluate the new model
    new_model_predictions = new_logistic_regression_model.predict(X_test_imputed)
    new_model_accuracy = accuracy_score(y_test, new_model_predictions)

    print("\nNew Model Accuracy:", new_model_accuracy)


"""
The output provides various metrics to evaluate the performance of the logistic regression model on the testing set. Here's an explanation of each part:

1. **Accuracy: 0.7872**
   - The accuracy represents the overall correctness of the model on the test dataset. In this case, the model achieves an accuracy of approximately 78.72%, indicating that it correctly predicts the target variable (death or survival) for about 78.72% of the characters.

2. **Classification Report:**
   - **Precision:** Precision is the ratio of correctly predicted positive observations to the total predicted positives. 
     - Precision for class 0 (characters who died) is 0.82, which means that when the model predicts a character will die, it is correct about 82% of the time.
     - Precision for class 1 (characters who survived) is 0.79, indicating that when the model predicts a character will survive, it is correct about 79% of the time.
   - **Recall (Sensitivity):** Recall is the ratio of correctly predicted positive observations to the all observations in actual class.
     - Recall for class 0 is 0.10, suggesting that the model identifies only 10% of the characters who actually died.
     - Recall for class 1 is 0.99, indicating that the model captures 99% of the characters who actually survived.
   - **F1-score:** The F1-score is the weighted average of precision and recall.
     - The F1-score for class 0 is 0.18, and for class 1, it is 0.88.
   - **Support:** The number of actual occurrences of the class in the specified dataset.

3. **Confusion Matrix:**
   - A confusion matrix is a table that describes the performance of a classification model. It consists of four values: true positive (298), true negative (9), false positive (81), and false negative (2).

4. **Coefficients:**
   - The coefficients represent the weights assigned to each feature by the logistic regression model. These coefficients indicate the impact of each feature on the likelihood of death.
     - `isNoble` has a positive impact (coefficient: 0.266883), suggesting that characters with noble status are more likely to survive.
     - `male` has a positive impact (coefficient: 0.250501), indicating that male characters are more likely to survive.
     - `age` has a very small positive impact (coefficient: 7.25e-06), suggesting a negligible effect.
     - `numDeadRelations` has a negative impact (coefficient: -0.317186), indicating that characters with more dead relations are less likely to survive.
     - `boolDeadRelations` has a negative impact (coefficient: -0.059259), suggesting that characters with dead relations are less likely to survive.
     - `isPopular` has a negative impact (coefficient: -0.032679), indicating that less popular characters are less likely to survive.
     - `popularity` has a small positive impact (coefficient: 0.004682), suggesting a minor effect.

5. **New Model Accuracy: 0.7872**
   - This is the accuracy of the model after fine-tuning. In this case, it appears that fine-tuning the model did not result in a significant change in accuracy.

Overall, the model performs reasonably well, but there are notable areas for improvement, especially in capturing characters who actually died (low recall for class 0). 
Depending on the specific goals and requirements, further adjustments and feature engineering may be necessary to enhance the model's performance.
   """