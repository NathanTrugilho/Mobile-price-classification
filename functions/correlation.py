import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def show_pearson_correlation(X, y, savePng=False):

    df_completo = pd.concat([X, y], axis=1)

    matriz_corr = df_completo.corr(method='pearson')

    plt.figure(figsize=(12, 10))

    mask = np.triu(np.ones_like(matriz_corr, dtype=bool), k=1)

    sns.heatmap(
        matriz_corr,
        mask=mask,
        annot=True,      # Mostra os valores de correlação nas células
        cmap='coolwarm', # Esquema de cores (quente para positivo, frio para negativo)
        fmt=".2f",       # Formata os números para duas casas decimais
        linewidths=.5
    )

    plt.tight_layout()

    if savePng:
        plt.savefig("correlation", dpi=300) 

    plt.show()

        