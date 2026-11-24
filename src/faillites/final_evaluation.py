"""Étude 2 — évaluation finale sur le jeu de test (utilisé une seule fois).

Même protocole que src.final_evaluation : candidats choisis en validation croisée
(meilleure dose de chaque méthode), réentraînés sur tout le développement avec 3 graines,
intervalles de confiance par bootstrap apparié, règle de parcimonie fixée à l'avance pour
le modèle retenu, calibration et seuil de coût appris hors-pli sur le développement.

Usage : python -m src.faillites.final_evaluation
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.faillites.config import FEATURES, FIG as FIG_DIR, MODEL_FILE, N_FOLDS, TAB as TAB_DIR, TARGET
from src.faillites.data import load_dev, load_test, split_xy
from src.faillites.features import Prep as Preprocessor
from src.faillites.generators import GENERATORS, get_synthetic
from src.faillites.tuning import load_params
from src.metrics import (COST_FN, COST_FP, MAIN_METRIC, at_threshold, best_f1, bootstrap_ci,
                         cost_threshold, calibrate, evaluate, fit_calibrator)
from src.models import make_classifier
from src.plots import FAMILIES, FAMILY_COLOR, GREY, METHOD_COLORS, set_style
from src.faillites.resampling import build_training_set
from src.resampling import n_to_add

PARAMS = load_params()

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2]
CLASSICAL = ["Sous-échantillonnage", "Duplication", "SMOTE"]
CLFS = ["LogReg", "XGBoost", "LightGBM"]


def parse_dose(d):
    if d in ("", None) or (isinstance(d, float) and np.isnan(d)):
        return None
    return d if d == "éq." else float(d)


def choose_candidates(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for clf in CLFS:
        g = best[best["classifieur"] == clf]
        keep = ["Aucune", "Pondération", g[g["méthode"].isin(CLASSICAL)].nlargest(1, MAIN_METRIC)["méthode"].iloc[0]]
        rows.append(g[g["méthode"].isin(keep + GENERATORS)])
    return pd.concat(rows, ignore_index=True)


def choose_final(cands: pd.DataFrame) -> pd.Series:
    top = cands.loc[cands[MAIN_METRIC].idxmax()]
    significant = top["méthode"] != "Aucune" and top["p-valeur (Holm)"] < 0.05
    if top["méthode"] == "Aucune" or significant:
        return top
    return cands[(cands["classifieur"] == top["classifieur"]) & (cands["méthode"] == "Aucune")].iloc[0]


def fit_predict(clf_kind, method, dose, X, y, prep, B, seed, tag):
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    synth = None
    if method in GENERATORS:
        synth = get_synthetic(method, f"{tag}_s{seed}", X[y == 1], n_to_add("éq.", n_pos, n_neg), seed)
    X_fit, y_fit = build_training_set(method, dose, X, y, prep, seed, synth)
    model = make_classifier(clf_kind, seed, weighted=(method == "Pondération"),
                            neg_pos_ratio=n_neg / n_pos, params=PARAMS[clf_kind]).fit(X_fit, y_fit)
    return model, model.predict_proba(B)[:, 1]


def oof_predictions(dev, clf_kind, method, dose) -> np.ndarray:
    """Prédictions hors-pli sur le jeu de développement (répétition 0)."""
    oof = np.zeros(len(dev))
    for k in range(N_FOLDS):
        tr, va = dev["fold_r0"] != k, dev["fold_r0"] == k
        Xtr, ytr = split_xy(dev[tr]); Xva, _ = split_xy(dev[va])
        prep = Preprocessor().fit(Xtr)
        _, oof[va.values] = fit_predict(clf_kind, method, dose, Xtr, ytr.values, prep,
                                        prep.transform(Xva), k, f"r0f{k}")
    return oof


def main() -> None:
    set_style()
    best = pd.read_csv(TAB_DIR / "cv_best.csv", keep_default_na=False)
    for c in [MAIN_METRIC, "p-valeur (Holm)", "gain"]:
        best[c] = pd.to_numeric(best[c], errors="coerce")
    cands = choose_candidates(best)
    final = choose_final(cands)
    print(f"Modèle final retenu d'après la validation croisée : {final['classifieur']} — {final['config']}\n")

    dev, test = load_dev(), load_test()
    Xdev, ydev = split_xy(dev); Xte, yte = split_xy(test)
    prep = Preprocessor().fit(Xdev)
    B, y, yt = prep.transform(Xte), ydev.values, yte.values
    print(f"Test : {len(yt)} demandes, {int(yt.sum())} défauts ({yt.mean():.1%})\n")

    rows, probas, final_model = [], {}, None
    for _, c in cands.iterrows():
        dose = parse_dose(c["dose"])
        per_seed, ps = [], []
        for s in SEEDS:
            model, p = fit_predict(c["classifieur"], c["méthode"], dose, Xdev, y, prep, B, s, "final")
            per_seed.append(evaluate(yt, p)); ps.append(p)
            if s == 0 and c["classifieur"] == final["classifieur"] and c["config"] == final["config"]:
                final_model = model
        probas[(c["classifieur"], c["config"])] = ps
        m = pd.DataFrame(per_seed)
        lo, hi = bootstrap_ci(yt, ps)
        row = {"classifieur": c["classifieur"], "config": c["config"], "méthode": c["méthode"],
               "PR-AUC validation croisée": c[MAIN_METRIC],
               "PR-AUC test": m[MAIN_METRIC].mean(), "écart-type (graines)": m[MAIN_METRIC].std(),
               "IC bas": lo, "IC haut": hi, "ROC-AUC test": m["ROC-AUC"].mean(),
               "KS test": m["KS"].mean(), "F1 max test": m["F1 max"].mean()}
        if c["méthode"] != "Aucune":
            ref = probas[(c["classifieur"], "Aucune")]
            dlo, dhi = bootstrap_ci(yt, ps, probas_ref=ref)
            diff = np.mean([average_precision_score(yt, p) for p in ps]) - np.mean(
                [average_precision_score(yt, p) for p in ref])
            row.update({"différence vs Aucune": diff, "diff. IC bas": dlo, "diff. IC haut": dhi})
        rows.append(row)
        print(f"{c['classifieur']:<9} {c['config']:<26} PR-AUC test = {row['PR-AUC test']:.3f} "
              f"[{lo:.3f} ; {hi:.3f}]", flush=True)

    results = pd.DataFrame(rows)
    results.round(4).to_csv(TAB_DIR / "test_results.csv", index=False)

    # --- Modèle final : seuils choisis sur le développement, appliqués au test -----------
    key = (final["classifieur"], final["config"])
    oof = oof_predictions(dev, final["classifieur"], final["méthode"], parse_dose(final["dose"]))
    p_final = probas[key][0]            # le modèle enregistré est celui de la graine 0
    # Calibration des probabilités, apprise sur les prédictions hors-pli du développement
    calibrator = fit_calibrator(y, oof)
    brier_raw = float(np.mean((p_final - yt) ** 2))
    oof, p_final = calibrate(calibrator, oof), calibrate(calibrator, p_final)
    brier_cal = float(np.mean((p_final - yt) ** 2))
    print(f"\nCalibration (Platt, apprise hors-pli) — score de Brier sur le test : "
          f"{brier_raw:.4f} avant, {brier_cal:.4f} après ; probabilité moyenne prédite {p_final.mean():.3f} "
          f"pour un taux de défaut réel de {yt.mean():.3f}")
    thr_cost = cost_threshold(y, oof)
    _, thr_f1 = best_f1(y, oof)
    decisions = []
    for name, t in [(f"Coût minimal ({COST_FN:.0f}:{COST_FP:.0f})", thr_cost), ("F1 maximal", thr_f1),
                    ("Seuil naïf", 0.5)]:
        decisions.append({"seuil choisi pour": name, "seuil": t, **at_threshold(yt, p_final, t),
                          "coût hors-pli par demande": at_threshold(y, oof, t)["Coût"] / len(y),
                          "coût test par demande": at_threshold(yt, p_final, t)["Coût"] / len(yt)})
    decisions = pd.DataFrame(decisions)
    decisions.round(4).to_csv(TAB_DIR / "test_decisions.csv", index=False)
    print("\nDécisions du modèle final sur le test, aux seuils fixés à l'avance :\n")
    print(decisions.round(3).to_string(index=False))
    print(f"\nPour comparaison, coût par demande sur le test : tout accepter = {COST_FN * yt.mean():.3f} ; "
          f"tout refuser = {COST_FP * (1 - yt.mean()):.3f}")

    joblib.dump({"model": final_model, "prep": prep, "calibrator": calibrator, "threshold": float(thr_cost),
                 "threshold_f1": float(thr_f1), "classifier": final["classifieur"],
                 "config": final["config"], "features": FEATURES}, MODEL_FILE)
    (TAB_DIR / "final_model.json").write_text(json.dumps({
        "classifieur": final["classifieur"], "config": final["config"],
        "seuil_cout": float(thr_cost), "brier_test_brut": brier_raw, "brier_test_calibre": brier_cal, "seuil_f1": float(thr_f1),
        "PR-AUC validation croisée": float(final[MAIN_METRIC])}, indent=2, ensure_ascii=False), encoding="utf-8")
    out = Xte.copy(); out[TARGET] = yt; out["proba"] = p_final
    out.to_csv(TAB_DIR / "test_predictions.csv", index=False)
    np.save(TAB_DIR / "oof_final.npy", oof)

    # --- Figures --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), sharex=True)
    for ax, clf in zip(axes, CLFS):
        d = results[results["classifieur"] == clf].iloc[::-1].reset_index(drop=True)
        for i, r in d.iterrows():
            color = FAMILY_COLOR[r["méthode"]]
            ax.plot([r["IC bas"], r["IC haut"]], [i, i], color=color, lw=2, solid_capstyle="round")
            ax.plot(r["PR-AUC test"], i, "o", color=color, ms=8, markeredgecolor="white", markeredgewidth=1.5)
        ref = d[d["méthode"] == "Aucune"]["PR-AUC test"].iloc[0]
        ax.axvline(ref, color=GREY, ls="--", lw=1)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d["config"])
        ax.set_title(clf); ax.set_xlabel("PR-AUC sur le test (IC à 95 %)")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, ms=7, markeredgecolor="white", label=name)
               for name, (c, _) in FAMILIES.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Évaluation finale sur le jeu de test (pointillés : sans rééquilibrage)", y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "test_pr_auc.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for _, r in results[results["classifieur"] == final["classifieur"]].iterrows():
        prec, rec, _ = precision_recall_curve(yt, probas[(r["classifieur"], r["config"])][0])
        ax.plot(rec, prec, lw=2.2 if r["méthode"] == "Aucune" else 1.4,
                ls="--" if r["méthode"] == "Aucune" else "-",
                color=METHOD_COLORS.get(r["méthode"], "#e87ba4"),
                label=f"{r['config']} ({r['PR-AUC test']:.3f})")
    ax.axhline(yt.mean(), color="#bbb", lw=1); ax.text(0.01, yt.mean() + 0.015, "hasard", fontsize=8, color="#888")
    ax.set_xlabel("Rappel (part des faillites détectées)"); ax.set_ylabel("Précision (part de refus justifiés)")
    ax.set_title(f"Courbes précision-rappel sur le test — {final['classifieur']}")
    ax.legend(fontsize=8, title="Méthode (PR-AUC)", title_fontsize=8, loc="lower left")
    fig.savefig(FIG_DIR / "test_courbes_pr.png"); plt.close(fig)


if __name__ == "__main__":
    main()
