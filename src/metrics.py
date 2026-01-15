import numpy as np


def calculate_edit_distance(ground_truth_text: str, predicted_text: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    rows = len(ground_truth_text) + 1
    cols = len(predicted_text) + 1
    distance_mat = np.zeros((rows, cols), dtype=int)

    distance_mat[:, 0] = np.arange(rows)
    distance_mat[0, :] = np.arange(cols)

    for i in range(1, rows):
        for j in range(1, cols):
            if ground_truth_text[i - 1] == predicted_text[j - 1]:
                distance_mat[i, j] = distance_mat[i - 1, j - 1]
            else:
                insert_cost = distance_mat[i, j - 1]
                delete_cost = distance_mat[i - 1, j]
                replace_cost = distance_mat[i - 1, j - 1]
                distance_mat[i, j] = min(insert_cost, delete_cost, replace_cost) + 1

    return int(distance_mat[rows - 1, cols - 1])


def calculate_relative_edit_distance(ground_truth_text: str, predicted_text: str) -> float:
    """Calculate relative Levenshtein edit distance between two strings. Result is non-negative value: 0 means the strings
    are identical, 1 - the strings are completely different. The result may also exceed 1 if predicted_text is much longer
    than ground_truth_text"""
    edit_distance = calculate_edit_distance(ground_truth_text, predicted_text)
    return edit_distance / len(ground_truth_text)
