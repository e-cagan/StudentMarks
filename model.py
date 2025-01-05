import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read the data
df = pd.read_csv('data/Student_Marks.csv')

# Handle missing data (if any)
df.dropna(inplace=True)

# Setting up the parameters
X = df[['number_courses', 'time_study']]
y = df[['marks']]

# Split the data to testing and training parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Predicting the mark
test_prediction = model.predict(X_test)

# Calculating r2 score and mean squared error for our model
r2 = r2_score(y_test, test_prediction)
mse = mean_squared_error(y_test, test_prediction)

# Main function for simulation of predicting student marks
def main():
    """Main simulation of student marks predictor."""

    while True:
        # Trying to take appropriate user inputs
        try:
            # Taking user inputs
            courses_taken = int(input('Enter number of courses taken: '))
            time_studied = int(input('Enter time studied (in hours): '))

            # Preparing the data for processing
            prep_array = np.array([[courses_taken, time_studied]])

            # Predicting the mark
            prediction = model.predict(prep_array)

            # Displaying the output
            print(f"Your mark should be {prediction[0][0]:.2f}.")
            print()

            # Displaying model performance
            print("Model performance")
            print(f"R2 Score: {r2:.2f}")
            print(f"Mean squared error: {mse:.2f}")
            print()
            break
        except ValueError:
            print("Please enter appropriate values in order to calculate correctly.")

    # Taking user input to continue using StudentMarks
    user_continue = input("Do you want to try again? (Type 'yes' for yes.): ").lower()
    if user_continue == 'yes':
        main()
    else:
        print("Thanks for using StudentMarks :)")

if __name__ == '__main__':
    main()
