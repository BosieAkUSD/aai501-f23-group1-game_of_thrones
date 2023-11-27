""" 
1.  Load Datasets: 
   - Use pandas to load the datasets (`battles.csv`, `character-deaths.csv`, `character-predictions.csv`).

2.  Explore and Preprocess Data: 
   - Use pandas to explore the structure of each dataset.
   - Handle any missing values in the datasets.
   - Merge or join relevant information from these datasets based on common keys.

   ^^^ ABOVE WILL BE COMPLETED BY @JABALI in EDA process

3.  Feature Selection: 
   - Identify relevant features for  our logistic regression model, such as age, nobility status, and gender.
   - Convert categorical variables into numerical format using techniques like one-hot encoding.

4.  Define Target Variable: 
   - Determine  our target variable, which in this case is whether a character died or survived.

5.  Split Data into Training and Testing Sets: 
   - Use scikit-learn to split the data into training and testing sets. This helps evaluate  our model's performance.

6.  Build Logistic Regression Model: 
   - Utilize the scikit-learn library to create a logistic regression model.
   - Train the model using the training dataset.

7.  Evaluate Model: 
   - Assess the model's performance on the testing dataset using metrics like accuracy, precision, recall, and F1 score.
   - Analyze the confusion matrix to understand the model's predictions.

8.  Interpret Coefficients: 
   - Examine the coefficients of the logistic regression model to interpret the impact of each feature on the likelihood of death.

9.  Fine-tune Model: 
   - Adjust model parameters or select different features based on the initial performance.
   - Reevaluate the model's performance and iterate if necessary.

10.  Save Code in a Python Script: 
    - Open a text editor or IDE, copy the Python code, and save it in a Python script file (e.g., `game_of_thrones_logistic_regression.py`).

11.  Run Script and Save Output: 
    - Execute the script using the Python interpreter in the command line or terminal.
    - Optionally, save the script output to a file using the command-line redirection, for example:
      ```bash
      python game_of_thrones_logistic_regression.py > output.txt
      ```

"""