import numpy as np
from sklearn.model_selection import KFold

from .model import Model

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns the F1 score of the given predictions.

    Args:
        y_true (np.ndarray): the true labels.
        y_pred (np.ndarray): the predicted labels.

    Returns:
        float: the F1 score of the given predictions.
    """
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    
    return 2 * (precision * recall) / (precision + recall)

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns the accuracy of the given predictions.

    Args:
        y_true (np.ndarray): the true labels.
        y_pred (np.ndarray): the predicted labels.

    Returns:
        float: the accuracy of the given predictions.
    """
    return np.mean(y_true == y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns the precision of the given predictions.

    Args:
        y_true (np.ndarray): the true labels.
        y_pred (np.ndarray): the predicted labels.

    Returns:
        float: the precision of the given predictions.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    
    return true_positives / (true_positives + false_positives)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns the recall of the given predictions.

    Args:
        y_true (np.ndarray): the true labels.
        y_pred (np.ndarray): the predicted labels.

    Returns:
        float: the recall of the given predictions.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    return true_positives / (true_positives + false_negatives)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns the F1 score of the given predictions.

    Args:
        y_true (np.ndarray): the true labels.
        y_pred (np.ndarray): the predicted labels.

    Returns:
        float: the F1 score of the given predictions.
    """
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    
    return 2 * (precision_val * recall_val) / (precision_val + recall_val)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns the confusion matrix of the given predictions.

    Args:
        y_true (np.ndarray): the true labels.
        y_pred (np.ndarray): the predicted labels.

    Returns:
        np.ndarray: the confusion matrix of the given predictions.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    confusion_matrix = np.zeros((2, 2))
    
    for i in range(2):
        for j in range(2):
            confusion_matrix[i, j] = np.sum((y_true == i) & (y_pred == j))
    
    return confusion_matrix

def cross_validate(model: Model, X: np.ndarray, y: np.ndarray):
    """
    cross validates the model with 10 folds and calculates the mean metrics.

    args:
        model (Model): the model to be cross validated.
        X (np.ndarray): the test data.
        y (np.ndarray): the test labels.
    
    """
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    accuracies = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy_score = accuracy(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall_score = recall(y_test, y_pred)
        precision_score = precision(y_test, y_pred)

        accuracies.append(accuracy_score)
        f1_scores.append(f1)
        recall_scores.append(recall_score)
        precision_scores.append(precision_score)
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    mean_recall = np.mean(recall_scores)
    std_recall = np.std(recall_scores)

    mean_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)
    
    print(f'Média Acurácia: {mean_accuracy:.4f}, Desvio Padrão: {std_accuracy:.4f}')
    print(f'Média F1-Score: {mean_f1:.4f}, Desvio Padrão: {std_f1:.4f}')
    print(f'Média Recall: {mean_recall:.4f}, Desvio Padrão: {std_recall:.4f}')
    print(f'Média Precisão: {mean_precision:.4f}, Desvio Padrão: {std_precision:.4f}')
