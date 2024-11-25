import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def generate_confusion_matrix_sklearn(actual_labels, predicted_labels, model_name = None):
    cm = confusion_matrix(actual_labels, predicted_labels)
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    display_cm.plot()
    if model_name:
        plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

def generate_confusion_matrix_custom(actual_labels, predicted_labels, model_name = None):
    n_classes = len(np.unique(actual_labels))
    result = np.zeros((n_classes, n_classes))

    for i in range(len(actual_labels)):

        result[actual_labels[i]][predicted_labels[i]] += 1

    if model_name:
        print(f"Confusion Matrix for {model_name}:")
    
    print("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    
    print(result)

# prototype for generating table
def generate_evaluation_table(models, test_features, test_labels, model_names):
