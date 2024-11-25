import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import pandas as pd

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
    
    print(result)
    print("Legend\n")
    print("Row Indecies from 0-9: ","airplane=0", "automobile=1", "bird=2", "cat=3", "deer=4", "dog=5", "frog=6", "horse=7", "ship=8", "truck=9")
    print("Column Indecies from 0-9: ","airplane=0", "automobile=1", "bird=2", "cat=3", "deer=4", "dog=5", "frog=6", "horse=7", "ship=8", "truck=9")


# helper method which generates a classification report
def extract_evaluation_metrics(actual_labels, predicted_labels, model_name = None):
    
    report = classification_report(actual_labels, predicted_labels, output_dict=True)
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    return [model_name, accuracy, precision, recall, f1]

def generate_evaluation_table(all_evaluation_metrics):
    columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Measure"]
    df = pd.DataFrame(all_evaluation_metrics, columns=columns)
    print(df.to_string(index=False))

    
