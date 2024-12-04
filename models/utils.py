import numpy as np

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
