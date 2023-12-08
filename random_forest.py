import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def random_forest_function():
    # Load datasets
    character_predictions = pd.read_csv("data_files/character-predictions.csv")
    character_deaths = pd.read_csv("data_files/character-deaths.csv")

    # Combine datasets based on common keys
    merged_data = pd.merge(character_predictions, character_deaths, how='left', left_on="name", right_on='Name')

    # Identify features and target variable
    selected_features = ['isNoble', 'male', 'age', 'numDeadRelations', 'boolDeadRelations', 'isPopular', 'popularity']
    X = merged_data[selected_features]
    y = merged_data['actual']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=1000, random_state=0)

    # Train the model with imputed data
    rf_model.fit(X_train_imputed, y_train)

    # Make predictions on the test set with imputed data
    y_pred = rf_model.predict(X_test_imputed)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{classification_rep}")