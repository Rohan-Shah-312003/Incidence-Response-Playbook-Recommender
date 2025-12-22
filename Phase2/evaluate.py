import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from train_models import train

os.makedirs("./Phase2/results", exist_ok=True)

def evaluate():
    models, X_test, y_test = train()

    for name, pipeline in models.items():
        print(f"\n=== {name} ===")

        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=pipeline.classes_)
        disp.plot(xticks_rotation=45)
        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"./Phase2/results/{name}_confusion_matrix.png")
        plt.close()

if __name__ == "__main__":
    evaluate()
