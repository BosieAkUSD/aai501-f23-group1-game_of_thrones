# main.py

# Import functions from other Python files
from game_of_thrones_logistic_regression import logistic_regression_analysis
#from random_forest import random_forest_function
from gradient_boosting import gradient_boosting_function
from exploratory_data import exploratory_data_analysis

def main():
    # Call functions from other files
    exploratory_data_analysis()
    logistic_regression_analysis()
    # random_forest_function()
    gradient_boosting_function()

if __name__ == "__main__":
    main()
