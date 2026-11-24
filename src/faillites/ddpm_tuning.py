"""Étude 2 — durée d'entraînement de TabDDPM : fidélité contre mémorisation.

Même règle, fixée à l'avance, que pour l'étude 1 : on retient le nombre d'itérations qui
donne le plus petit écart de distribution parmi ceux dont le rapport DCR reste ≥ 0,8.
Le critère ne regarde que la qualité des données, jamais la performance des classifieurs.

Usage : python -m src.faillites.ddpm_tuning --folds 0     (un pli par processus)
        python -m src.faillites.ddpm_tuning --analyse-only
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd

from src.faillites.config import FIG, TAB
from src.faillites.data import load_dev, split_xy
from src.faillites.features import Prep
from src.faillites.generators import DDPM_DEFAULT, DDPM_FILE, fit_generate
from src.faillites.quality import quality_report
from src.plots import AQUA, ORANGE, set_style

ITERS = [1000, 2500, 5000]  # au-delà, l’écart et la mémorisation ne font que croître (tendance monotone)
DCR_MIN = 0.8
N_GEN = 1000


def run(fold: int) -> None:
    out = TAB / f"ddpm_tuning_runs_f{fold}.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {r["itérations"] for r in rows}
    dev = load_dev()
    Xtr, ytr = split_xy(dev[dev["fold_r0"] != fold]); Xva, yva = split_xy(dev[dev["fold_r0"] == fold])
    prep = Prep().fit(Xtr)
    train_def, hold_def = Xtr[ytr.values == 1], Xva[yva.values == 1]
    for it in ITERS:
        if it in done:
            continue
        t0 = time.time()
        fake = fit_generate("TabDDPM", train_def, N_GEN, seed=fold, ddpm={**DDPM_DEFAULT, "n_iter": it})
        rows.append({"pli": fold, "itérations": it, "durée (s)": round(time.time() - t0),
                     **quality_report(train_def, hold_def, fake, prep)})
        pd.DataFrame(rows).round(4).to_csv(out, index=False)
        r = rows[-1]
        print(f"pli {fold} | {it:>6} it | écart {r['Écart de distribution']:.3f} | DCR {r['Rapport DCR']:.2f} "
              f"| quasi-copies {r['Quasi-copies (%)']:.1f} % | {r['durée (s)']} s", flush=True)


def analyse() -> None:
    set_style()
    runs = pd.concat([pd.read_csv(f) for f in sorted(TAB.glob("ddpm_tuning_runs_f*.csv"))])
    s = runs.groupby("itérations").mean(numeric_only=True).drop(columns="pli")
    s.round(4).to_csv(TAB / "ddpm_tuning.csv")
    ok = s[s["Rapport DCR"] >= DCR_MIN]
    best = int(ok["Écart de distribution"].idxmin())
    DDPM_FILE.write_text(json.dumps({**DDPM_DEFAULT, "n_iter": best}, indent=2), encoding="utf-8")
    print(s.round(3).to_string(), f"\n\nItérations retenues : {best}")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].plot(s.index, s["Écart de distribution"], "o-", color=AQUA)
    axes[0].set_title("Fidélité : écart de distribution\n(plus bas = mieux)")
    axes[1].plot(s.index, s["Rapport DCR"], "o-", color=ORANGE)
    axes[1].axhline(DCR_MIN, ls="--", color="#888", lw=1)
    axes[1].set_title("Mémorisation : rapport DCR\n(1 = comme un vrai inédit)")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks(ITERS); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xticklabels([f"{i:,}".replace(",", " ") for i in ITERS])
        ax.axvline(best, color=AQUA, alpha=0.3, lw=6); ax.set_xlabel("Itérations d'entraînement")
    fig.savefig(FIG / "ddpm_tuning.png"); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--analyse-only", action="store_true")
    a = p.parse_args()
    if not a.analyse_only:
        for f in a.folds:
            run(f)
    analyse()
