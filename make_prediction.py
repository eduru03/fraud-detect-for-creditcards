import joblib
import pandas as pd

# Load the saved model
model = joblib.load('best_model.pkl')

# Load new data or use part of the test data for prediction
new_data = pd.read_csv('creditcard.csv').drop('Class', axis=1).iloc[:5]  # Example

# Make predictions
predictions = model.predict(new_data)
print("Predictions:", predictions)
