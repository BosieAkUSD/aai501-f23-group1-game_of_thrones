from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier  # Changed to classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def gradient_boosting_function():
    print(" ")
    print("Running Gradient Boosting Analysis...")
    character_predictions = pd.read_csv("data_files/character-predictions.csv")
    character_deaths = pd.read_csv("data_files/character-deaths.csv")

    selected_features = ['isNoble', 'male', 'age', 'numDeadRelations', 'boolDeadRelations', 'isPopular', 'popularity']

    merged_data = pd.merge(character_predictions, character_deaths, how='left', left_on='name', right_on='Name')

    X = merged_data[selected_features]
    y = merged_data['actual']

    # Convert to classication
    y_binary = np.where(y > 0, 1, 0)

    # Imputing because some values are NaN/None
    imputer = SimpleImputer(strategy='mean') 
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_binary, test_size=0.2, random_state=42)

    gb_classifier = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000, max_depth=3, random_state=42)
    
    gb_classifier.fit(X_train, y_train)

    y_pred = gb_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{classification_rep}")

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')


    r2 = r2_score(y_test, y_pred)
    print(f'R^2 Score: {r2}')


    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')

    feature_importance = gb_classifier.feature_importances_
    feature_importance_dict = dict(zip(selected_features, feature_importance))
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    print('\nFeature Importance:')
    for feature, importance in sorted_feature_importance:
        print(f'{feature}: {importance}')
