from ml_functions import load_iris_data, preprocess_data, train_model, evaluate_model
from ml_model import create_logistic_regression_model
import pickle

def main():
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    
    # Save the trained model as a pickle file
    with open('iris_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Prompt user to input new data for testing
    print("\nInput new data for testing:")
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))

    # Make prediction based on user input
    new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(new_data)
    predicted_class = prediction[0]
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_species = species_names[predicted_class]

    print(f"\nPredicted Species: {predicted_species}")

if __name__ == "__main__":
    main()
