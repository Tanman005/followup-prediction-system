import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_model_accuracy(results):
    models = list(results.keys())
    accuracies = list(results.values())

    plt.figure()
    plt.bar(models, accuracies)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    plt.xticks(rotation=20)
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()