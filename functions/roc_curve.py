import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def calculate_roc_curve(y_true, y_pred_proba):

    classes = np.unique(y_true)
    n_classes = len(classes)
    
    y_true_binarized = label_binarize(y_true, classes=classes)
    
    results = {}
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        class_label = classes[i]
        
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        results[class_label] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
        
        plt.plot(fpr, tpr, label=f'ROC Classe {class_label} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC para Cada Classe')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curve", dpi=300)
    plt.show()
    
    # Comentei pq não vou usar o dicionário com os valores
    # return results
