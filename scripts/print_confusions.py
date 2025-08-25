import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
LABELS=['O','B-FIELD','I-FIELD','B-NUM','I-NUM','B-HEADER','I-HEADER']

def show_and_plot(name: str):
    p = os.path.join(OUTDIR, f'{name}_confusion.npy')
    if not os.path.exists(p):
        print(f'Confusion file not found: {p}')
        return
    cm = np.load(p)
    print(f"\n{name} confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    fig,ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha='right')
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{name} Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i,j]), ha='center', va='center', color='black', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{name}_confusion.png'), dpi=150)
    plt.close(fig)

if __name__ == '__main__':
    show_and_plot('layoutlm')
    show_and_plot('bilstm_crf')
