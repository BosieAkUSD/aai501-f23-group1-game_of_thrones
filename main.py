# main.py

# Import functions from other Python files
from game_of_thrones_logistic_regression import logistic_regression_function
from random_forest import random_forest_function
from gradient_boosting import gradient_boosting_function
from exploratory_data import exploratory_data_function

def main():
    # Call functions from other files
    logistic_regression_function()
    random_forest_function()
    gradient_boosting_function()
    exploratory_data_function()

if __name__ == "__main__":
    main()
