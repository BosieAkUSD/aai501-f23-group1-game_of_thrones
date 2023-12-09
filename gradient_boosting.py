from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def gradient_boosting_function():
    character_predictions = pd.read_csv("data_files/character-predictions.csv")
    character_deaths = pd.read_csv("data_files/character-deaths.csv")

    selected_features = ['isNoble', 'male', 'age', 'numDeadRelations', 'boolDeadRelations', 'isPopular', 'popularity']

    merged_data = pd.merge(character_predictions, character_deaths, how='left', left_on='name', right_on='Name')

    X = merged_data[selected_features]
    y = merged_data['actual']

    # Imputing because some values are NaN/None
    imputer = SimpleImputer(strategy='mean') 
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    gb_regressor = GradientBoostingRegressor(learning_rate=0.1, n_estimators=1000, max_depth=3, random_state=42)
    
    gb_regressor.fit(X_train, y_train)

    y_pred = gb_regressor.predict(X_test)


    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')


    r2 = r2_score(y_test, y_pred)
    print(f'R^2 Score: {r2}')


    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')


    feature_importance = gb_regressor.feature_importances_
    feature_importance_dict = dict(zip(selected_features, feature_importance))
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    print('\nFeature Importance:')
    for feature, importance in sorted_feature_importance:
        print(f'{feature}: {importance}')
