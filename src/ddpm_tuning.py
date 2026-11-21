"""Réglage de TabDDPM : compromis entre fidélité et mémorisation.

Avec environ 760 défauts pour apprendre, la durée d'entraînement de TabDDPM est un réglage
délicat :
- trop court : le modèle n'a pas convergé, les clients générés sont peu fidèles (valeurs
  collées aux bornes, liens entre variables mal reproduits) ;
- trop long : le modèle mémorise les vrais clients et les recopie presque à l'identique,
  ce qui n'apporte aucune information nouvelle au classifieur et poserait un problème de
  confidentialité.

Ce script entraîne TabDDPM avec différentes durées, sur deux plis du jeu de développement,
et mesure à chaque fois la fidélité et la mémorisation (voir src.quality).

RÈGLE DE CHOIX, fixée sans regarder l'utilité pour le classifieur :
parmi les réglages dont le rapport DCR reste au moins égal à DCR_MIN (les faux clients ne
sont pas nettement plus proches des données d'entraînement que de vrais clients inédits),
on retient celui dont l'écart de distribution (moyenne du KS et de la TVD) est le plus
faible. Le réglage retenu est écrit dans reports/tables/ddpm_config.json et utilisé partout.

À titre d'information seulement, le script mesure aussi le gain de PR-AUC obtenu en
ajoutant un défaut synthétique par vrai défaut : cela permet de vérifier que la conclusion
du projet ne dépend pas de ce réglage.

Le jeu de test n'est jamais utilisé ici.

Usage :
    python -m src.ddpm_tuning                 # calculs puis analyse
    python -m src.ddpm_tuning --folds 0       # un seul pli (calcul en parallèle)
    python -m src.ddpm_tuning --analyse-only
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIG_DIR, TAB_DIR
from src.features import Preprocessor, load_dev, split_xy
from src.generators import DDPM_DEFAULT, DDPM_FILE, fit_generate
from src.metrics import evaluate
from src.models import make_classifier
from src.plots import BLUE, RED, set_style
from src.quality import quality_report
from src.resampling import build_training_set

ITERS = [1000, 2500, 5000, 10000, 20000]
FOLDS = [0, 1]
DCR_MIN = 0.8             # 1 = aucune mémorisation, 0 = copies
N_SAMPLES = 2000


def run_fold(dev: pd.DataFrame, fold: int) -> None:
    out = TAB_DIR / f"ddpm_tuning_runs_f{fold}.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {r["itérations"] for r in rows}
    Xtr, ytr = split_xy(dev[dev["fold_r0"] != fold])
    Xva, yva = split_xy(dev[dev["fold_r0"] == fold])
    y = ytr.values
    train, holdout = Xtr[y == 1], Xva[yva.values == 1]
    prep = Preprocessor().fit(Xtr)
    B = prep.transform(Xva)

    def utility(method, synth=None):
        X_fit, y_fit = build_training_set(method, 1, Xtr, y, prep, fold, synth)
        return {kind: evaluate(yva.values, make_classifier(kind, fold).fit(X_fit, y_fit)
                               .predict_proba(B)[:, 1])["PR-AUC"] for kind in ("LogReg", "XGBoost")}

    base = utility("Aucune")
    for n_iter in ITERS:
        if n_iter in done:
            continue
        t0 = time.time()
        fake = fit_generate("TabDDPM", train, N_SAMPLES, seed=fold, ddpm={**DDPM_DEFAULT, "n_iter": n_iter})
        res = quality_report(train, holdout, fake, prep)
        res.update({f"Gain PR-AUC {k}": v - base[k] for k, v in utility("TabDDPM", fake).items()})
        rows.append({"pli": fold, "itérations": n_iter, "durée (s)": round(time.time() - t0), **res})
        pd.DataFrame(rows).round(4).to_csv(out, index=False)
        print(f"pli {fold} | {n_iter:>5} it. | KS {res['KS moyen']:.3f} | TVD {res['TVD moyenne']:.3f} | "
              f"rapport DCR {res['Rapport DCR']:.2f} | quasi-copies {res['Quasi-copies (%)']:.0f} %", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    set_style()
    if not args.analyse_only:
        dev = load_dev()
        for fold in args.folds:
            run_fold(dev, fold)

    runs = pd.concat([pd.read_csv(f) for f in sorted(TAB_DIR.glob("ddpm_tuning_runs_f*.csv"))])
    summary = runs.groupby("itérations").mean(numeric_only=True).drop(columns="pli")
    summary["Écart de distribution"] = (summary["KS moyen"] + summary["TVD moyenne"]) / 2
    summary.round(3).to_csv(TAB_DIR / "ddpm_tuning.csv")
    print(f"\n{runs['pli'].nunique()} plis.\n", summary.round(3).to_string())

    ok = summary[summary["Rapport DCR"] >= DCR_MIN]
    n_iter = int(ok["Écart de distribution"].idxmin())
    cfg = {**DDPM_DEFAULT, "n_iter": n_iter}
    DDPM_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nRéglage retenu : {cfg}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(summary.index, summary["KS moyen"], color=BLUE, marker="o", lw=2, ms=7,
                 markeredgecolor="white", markeredgewidth=1.5, label="KS moyen (variables numériques)")
    axes[0].plot(summary.index, summary["Écart corrélations"], color="#eb6834", marker="s", lw=2, ms=7,
                 markeredgecolor="white", markeredgewidth=1.5, label="Écart de corrélations")
    axes[0].set_title("Fidélité (plus bas = plus fidèle)"); axes[0].legend()
    axes[1].plot(summary.index, summary["Rapport DCR"], color=BLUE, marker="o", lw=2, ms=7,
                 markeredgecolor="white", markeredgewidth=1.5)
    axes[1].axhline(DCR_MIN, color=RED, ls=":", lw=1.2)
    axes[1].text(ITERS[0], DCR_MIN + 0.02, f"limite fixée : {DCR_MIN}", fontsize=8, color=RED)
    axes[1].set_title("Mémorisation : rapport DCR\n(1 = aucune mémorisation, 0 = copies)")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks(ITERS); ax.set_xticklabels([str(i) for i in ITERS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Itérations d'entraînement de TabDDPM")
        ax.axvline(n_iter, color="#8a8a8a", ls="--", lw=1)
    axes[0].text(n_iter, axes[0].get_ylim()[1], " réglage retenu", fontsize=8, color="#555", va="top")
    fig.tight_layout(); fig.savefig(FIG_DIR / "ddpm_tuning.png"); plt.close(fig)


if __name__ == "__main__":
    main()
