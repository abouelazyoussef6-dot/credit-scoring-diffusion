"""Construit et exécute notebooks/02_eda_faillites.ipynb (analyse exploratoire de l'étude 2).

Usage : python -m src.faillites.build_eda_notebook
"""
import subprocess

import nbformat as nbf

from src.config import ROOT

NB = ROOT / "notebooks" / "02_eda_faillites.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Étude 2 — Analyse exploratoire : faillites d'entreprises (fort déséquilibre)\n\n"
       "Données : *Polish Companies Bankruptcy* (UCI, Tomczak 2016, CC BY 4.0), fichier de la 3e année : "
       "64 ratios financiers d'une année et faillite dans les 3 ans qui suivent. Pour une banque, la faillite "
       "d'une entreprise emprunteuse est l'événement de défaut : c'est un problème de **scoring de crédit des "
       "entreprises**.\n\n**Toute l'analyse porte sur le jeu de développement** : le jeu de test a été mis de "
       "côté avant (`python -m src.faillites.data`)."),
    code("import sys, pathlib\nsys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
         "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
         "from scipy.stats import ks_2samp\n"
         "from src.faillites.config import FEATURES, LABELS, TARGET, FIG\n"
         "from src.faillites.data import load_dev\nfrom src.faillites.features import slog\n"
         "from src.plots import set_style, C_GOOD, C_BAD\nset_style()\n"
         "dev = load_dev(); X, y = dev[FEATURES], dev[TARGET]\n"
         "print(f'{len(dev)} entreprises, {y.sum()} faillites ({y.mean():.2%}) : 1 faillite pour {(1-y.mean())/y.mean():.0f} entreprises saines')"),
    md("## 1. Valeurs manquantes\n\nLa moitié des entreprises ont au moins un ratio manquant. Comme pour HMEQ, "
       "on vérifie si l'absence d'un ratio est liée à la faillite."),
    code("miss = X.isna().mean().sort_values(ascending=False)\n"
         "print(f\"Entreprises complètes : {X.notna().all(axis=1).mean():.1%}\")\n"
         "tab = pd.DataFrame({'manquant (%)': 100*miss, 'faillites si manquant (%)': [100*y[X[c].isna()].mean() if X[c].isna().any() else np.nan for c in miss.index],\n"
         "                    'faillites si renseigné (%)': [100*y[X[c].notna()].mean() for c in miss.index]})\n"
         "tab = tab[tab['manquant (%)'] >= 1]; display(tab.round(1))\n"
         "fig, ax = plt.subplots(figsize=(8, 3.4)); xs = np.arange(len(tab)); w = .38\n"
         "ax.bar(xs-w/2, tab['faillites si renseigné (%)'], w, color=C_GOOD, label='ratio renseigné')\n"
         "ax.bar(xs+w/2, tab['faillites si manquant (%)'], w, color=C_BAD, label='ratio manquant')\n"
         "ax.axhline(100*y.mean(), color='#888', ls='--', lw=1); ax.set_xticks(xs); ax.set_xticklabels(tab.index, fontsize=8)\n"
         "ax.set_ylabel('Taux de faillite (%)'); ax.set_title('Taux de faillite selon que le ratio est renseigné ou non'); ax.legend()\n"
         "fig.savefig(FIG/'eda_manquants.png'); plt.show()"),
    md("## 2. Des distributions à queues extrêmes\n\nUn ratio dont le dénominateur est proche de zéro explose : "
       "l'asymétrie est énorme. La transformation logarithmique signée s(x) = signe(x)·log(1+|x|) la ramène à "
       "un niveau raisonnable sans changer l'ordre des entreprises."),
    code("sk = X.skew(); sk2 = slog(X).skew()\n"
         "print(f'Asymétrie médiane : {sk.abs().median():.1f} (brute) -> {sk2.abs().median():.2f} (log signé)')\n"
         "print(f'Valeur absolue maximale observée : {np.nanmax(np.abs(X.values)):,.0f}')\n"
         "fig, axes = plt.subplots(1, 2, figsize=(9, 3))\n"
         "axes[0].hist(X['X27'].dropna(), bins=60, color=C_GOOD); axes[0].set_title('X27 brut'); axes[0].set_yscale('log')\n"
         "axes[1].hist(slog(X['X27'].dropna()), bins=60, color=C_GOOD); axes[1].set_title('X27 après log signé')\n"
         "fig.suptitle(LABELS['X27']); fig.tight_layout(); fig.savefig(FIG/'eda_asymetrie.png'); plt.show()"),
    md("## 3. Quels ratios séparent les faillites ?\n\nStatistique de Kolmogorov-Smirnov entre faillites et "
       "entreprises saines, ratio par ratio (0 = aucune différence, 1 = séparation parfaite)."),
    code("ks = pd.Series({c: ks_2samp(X.loc[y==1, c].dropna(), X.loc[y==0, c].dropna()).statistic for c in FEATURES}).sort_values(ascending=False)\n"
         "top = ks.head(12)\n"
         "display(pd.DataFrame({'KS': top.round(3), 'définition': [LABELS[c] for c in top.index]}))\n"
         "fig, axes = plt.subplots(2, 3, figsize=(12, 6))\n"
         "for ax, c in zip(axes.ravel(), top.index[:6]):\n"
         "    v = slog(X[c]); lo, hi = np.nanpercentile(v, [1, 99]); b = np.linspace(lo, hi, 40)\n"
         "    ax.hist(v[y==0].clip(lo, hi), bins=b, density=True, alpha=.55, color=C_GOOD, label='saines')\n"
         "    ax.hist(v[y==1].clip(lo, hi), bins=b, density=True, alpha=.55, color=C_BAD, label='faillites')\n"
         "    ax.set_title(f'{c} · KS {ks[c]:.2f}\\n{LABELS[c][:45]}', fontsize=9); ax.set_yticks([])\n"
         "axes[0,0].legend(); fig.suptitle('Les 6 ratios les plus discriminants (échelle log signée)'); fig.tight_layout()\n"
         "fig.savefig(FIG/'eda_ratios.png'); plt.show()"),
    md("## 4. Des ratios très redondants\n\nPlusieurs ratios partagent numérateur ou dénominateur. Un bon "
       "générateur doit reproduire cette structure."),
    code("corr = X.corr('spearman'); iu = np.triu_indices(64, 1); a = corr.values[iu]\n"
         "print(f'Paires de ratios avec |corrélation de Spearman| > 0,9 : {(np.abs(a) > .9).sum()} sur {len(a)}')\n"
         "fig, ax = plt.subplots(figsize=(5.2, 4.4)); im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)\n"
         "ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); fig.colorbar(im, label='Spearman')\n"
         "ax.set_title('Corrélations entre les 64 ratios'); fig.savefig(FIG/'eda_correlations.png'); plt.show()"),
    md("## 5. Ce que l'exploration impose à la suite\n\n"
       "| Constat | Conséquence |\n|---|---|\n"
       "| 4,7 % de faillites : 1 pour 20 entreprises saines | PR-AUC comme métrique principale ; doses de 1, 2, 5 et jusqu'à l'équilibre (×20) |\n"
       "| Seulement environ 315 faillites par pli d'entraînement | Les générateurs apprennent sur très peu d'exemples : le cas où la génération devrait le plus servir |\n"
       "| Ratios manquants, parfois liés à la faillite | Indicateur « manquant » ; les générateurs reproduisent les valeurs manquantes |\n"
       "| Asymétrie extrême (valeurs jusqu'à 10^7) | Log signé avant standardisation, et pour les générateurs |\n"
       "| Ratios fortement corrélés entre eux | La préservation des corrélations est un critère de qualité |\n\n"
       "Toutes ces statistiques sont calculées sur le jeu de développement."),
]

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, NB)
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NB)], check=True,
               cwd=NB.parent)
print("notebook exécuté :", NB)
