from sklearn.linear_model import LogisticRegression

def create_logistic_regression_model():
    model = LogisticRegression(max_iter=1000)
    return model
