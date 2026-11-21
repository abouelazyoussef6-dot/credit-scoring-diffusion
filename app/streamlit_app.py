"""Application de démonstration : scoring de crédit (jeu de données HMEQ).

Lancement local :   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import CAT_COLS, LABELS, NUM_COLS, RAW_COLS, TARGET
from src.features import CATEGORIES, FEATURES
from src.metrics import calibrate

APP_DATA = ROOT / "app" / "data"
TAB, FIG = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
C_GOOD, C_BAD = "#2a78d6", "#eb6834"
JOB_LABELS = {"Mgr": "Cadre dirigeant", "Office": "Employé de bureau", "Other": "Autre",
              "ProfExe": "Profession libérale / cadre", "Sales": "Commercial", "Self": "Indépendant"}
REASON_LABELS = {"DebtCon": "Regroupement de dettes", "HomeImp": "Travaux dans le logement"}
UNKNOWN = "Non renseigné"

st.set_page_config(page_title="Scoring de crédit — données synthétiques", page_icon="🏦", layout="wide")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "final_model.joblib")


@st.cache_data
def load_csv(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


@st.cache_resource
def get_explainer():
    b = load_model()
    if hasattr(b["model"], "coef_"):
        return shap.LinearExplainer(b["model"], b["prep"].transform(load_csv(str(APP_DATA / "demo_clients.csv"))[RAW_COLS]))
    return shap.TreeExplainer(b["model"])


def score(raw: pd.DataFrame) -> np.ndarray:
    b = load_model()
    return calibrate(b["calibrator"], b["model"].predict_proba(b["prep"].transform(raw[RAW_COLS]))[:, 1])


def explain(raw_row: pd.DataFrame) -> shap.Explanation:
    """Explication SHAP d'une demande, avec des valeurs affichées dans leurs unités d'origine."""
    b = load_model()
    X = b["prep"].transform(raw_row[RAW_COLS])
    exp = get_explainer()(X)
    values = exp.values[0, :, 1] if exp.values.ndim == 3 else exp.values[0]
    base = np.ravel(exp.base_values)[-1 if exp.values.ndim == 3 else 0]
    display = pd.Series(X[0], index=FEATURES)
    for c in NUM_COLS:
        display[c] = raw_row[c].iloc[0]
    display["LTV"] = (raw_row["LOAN"].iloc[0] + raw_row["MORTDUE"].iloc[0]) / raw_row["VALUE"].iloc[0]
    return shap.Explanation(values=values, base_values=base, data=display.round(2).values, feature_names=FEATURES)


def readable(feature: str, value: float) -> str:
    """Nom lisible d'une colonne du modèle, compte tenu de sa valeur pour ce dossier."""
    if feature.endswith("_manquant"):
        return f"{LABELS[feature[:-9]].lower()} {'non renseigné' if value >= 0.5 else 'renseigné'}"
    if "=" in feature:
        var, val = feature.split("=")
        label = ({**JOB_LABELS, **REASON_LABELS}).get(val, UNKNOWN).lower()
        return f"{LABELS[var].lower()} : {label}" if value >= 0.5 else f"{LABELS[var].lower()} autre que « {label} »"
    if feature == "LTV":
        return f"endettement sur le bien ({value:.0%} de sa valeur)"
    return f"{LABELS.get(feature, feature).lower()} ({value:g})"


bundle = load_model()
demo = load_csv(str(APP_DATA / "demo_clients.csv"))
scores = load_csv(str(APP_DATA / "test_scores.csv"))

st.title("Scoring de crédit : estimer le risque de défaut d'une demande de prêt")
suffix = "sans rééquilibrage des classes" if bundle["config"] == "Aucune" else bundle["config"]
st.caption(f"Modèle : {bundle['classifier']}, {suffix} — données HMEQ (prêts garantis par un bien immobilier)")

tab_form, tab_thr, tab_cmp, tab_syn, tab_csv, tab_about = st.tabs([
    "Évaluer une demande", "Régler le seuil de décision", "Comparaison des méthodes",
    "Défauts synthétiques", "Scorer un fichier", "À propos"])

# ------------------------------------------------------------------------------------------
with tab_form:
    left, right = st.columns([5, 6], gap="large")
    with left:
        options = ["Nouveau dossier"] + [f"Dossier n° {i} du jeu de test" for i in demo.index]
        choice = st.selectbox("Point de départ", options,
                              help="Partir d'un dossier réel du jeu de test (jamais vu à l'entraînement) ou d'un dossier vierge.")
        if choice == "Nouveau dossier":
            start = demo[RAW_COLS].median(numeric_only=True).to_dict() | {"REASON": "DebtCon", "JOB": "Other"}
            truth, key = None, "new"
        else:
            i = int(choice.split("n° ")[1].split(" ")[0])
            start, truth, key = demo.loc[i, RAW_COLS].to_dict(), int(demo.loc[i, TARGET]), f"c{i}"

        st.markdown("**Le prêt et le bien**")
        values = {}
        c1, c2 = st.columns(2)
        values["LOAN"] = c1.number_input(LABELS["LOAN"], 1000, 100000, int(start["LOAN"]), 500, key=f"LOAN{key}")
        reasons = list(REASON_LABELS) + [None]
        r0 = start["REASON"] if start["REASON"] in REASON_LABELS else None
        values["REASON"] = c2.selectbox(LABELS["REASON"], reasons, index=reasons.index(r0), key=f"REASON{key}",
                                        format_func=lambda v: REASON_LABELS.get(v, UNKNOWN))

        def number(col, container, lo, hi, step, fmt=None):
            known = not pd.isna(start[col])
            default = float(start[col]) if known else float(demo[col].median())
            v = container.number_input(LABELS[col], float(lo), float(hi), float(np.clip(default, lo, hi)), float(step),
                                       key=f"{col}{key}", format=fmt)
            missing = container.checkbox(UNKNOWN, value=not known, key=f"{col}_na{key}")
            values[col] = np.nan if missing else v

        c1, c2 = st.columns(2)
        number("VALUE", c1, 5000, 900000, 5000, "%.0f")
        number("MORTDUE", c2, 0, 400000, 5000, "%.0f")
        st.markdown("**Le demandeur**")
        c1, c2, c3 = st.columns(3)
        jobs = list(JOB_LABELS) + [None]
        j0 = start["JOB"] if start["JOB"] in JOB_LABELS else None
        values["JOB"] = c1.selectbox(LABELS["JOB"], jobs, index=jobs.index(j0), key=f"JOB{key}",
                                     format_func=lambda v: JOB_LABELS.get(v, UNKNOWN))
        number("YOJ", c2, 0, 45, 1, "%.0f")
        number("DEBTINC", c3, 0, 100, 1, "%.1f")
        st.markdown("**Son historique de crédit**")
        c1, c2, c3 = st.columns(3)
        number("DELINQ", c1, 0, 15, 1, "%.0f")
        number("DEROG", c2, 0, 10, 1, "%.0f")
        number("NINQ", c3, 0, 17, 1, "%.0f")
        c1, c2, c3 = st.columns(3)
        number("CLAGE", c1, 0, 600, 12, "%.0f")
        number("CLNO", c2, 0, 71, 1, "%.0f")

    with right:
        row = pd.DataFrame([values])[RAW_COLS]
        p, thr = float(score(row)[0]), bundle["threshold"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilité de défaut estimée", f"{p:.1%}")
        c2.metric("Décision proposée", "Refus" if p >= thr else "Accord", help=f"Seuil de décision : {thr:.3f}")
        if truth is not None:
            c3.metric("Issue réelle du dossier d'origine", "Défaut" if truth == 1 else "Remboursé")
        st.progress(min(p, 1.0))

        e = explain(row)
        contrib = pd.Series(e.values, index=FEATURES).sort_values()
        shown = pd.Series(e.data, index=FEATURES)
        up = [readable(f, shown[f]) for f in contrib.index[::-1][:3] if contrib[f] > 0.05 and not pd.isna(shown[f])]
        down = [readable(f, shown[f]) for f in contrib.index[:3] if contrib[f] < -0.05 and not pd.isna(shown[f])]
        if up:
            st.markdown("**Ce qui augmente le risque estimé :** " + " ; ".join(up) + ".")
        if down:
            st.markdown("**Ce qui le réduit :** " + " ; ".join(down) + ".")
        st.caption("Détail des contributions (valeurs SHAP). Rouge : pousse vers le défaut. Bleu : pousse vers le "
                   "remboursement. Les montants sont en unités d'origine ; « _manquant = 1 » signifie que "
                   "l'information n'est pas renseignée. L'axe est celui du score brut du modèle (log-cote), "
                   "avant calibration en probabilité.")
        plt.figure()
        shap.plots.waterfall(e, max_display=9, show=False)
        for _ax in plt.gcf().axes:
            _labels = [t.get_text() for t in _ax.get_yticklabels()]
            if any("other features" in t for t in _labels):
                _ax.set_yticks(_ax.get_yticks())
                _ax.set_yticklabels([t.replace("other features", "autres colonnes") for t in _labels], fontsize=13)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Outil pédagogique : une décision de crédit réelle ne repose jamais sur un score seul.")

# ------------------------------------------------------------------------------------------
with tab_thr:
    st.markdown("Le modèle produit une probabilité ; c'est le **seuil** qui la transforme en décision. Un seuil bas "
                "refuse plus de futurs défaillants, mais aussi plus de bons payeurs. Le bon réglage dépend du coût "
                "relatif des deux erreurs.")
    y, pr = scores[TARGET].values, scores["proba"].values
    c1, c2 = st.columns(2)
    thr = c1.slider("Seuil de décision (refus si la probabilité dépasse le seuil)", 0.005, 0.995,
                    float(round(bundle["threshold"], 3)), 0.001, format="%.3f")
    ratio = c2.slider("Coût d'un défaut accepté, en multiples du coût d'un bon payeur refusé", 1.0, 20.0, 5.0, 0.5)
    refuse = pr >= thr
    tp, fp, fn = int((refuse & (y == 1)).sum()), int((refuse & (y == 0)).sum()), int((~refuse & (y == 1)).sum())
    c = st.columns(5)
    c[0].metric("Défauts refusés", f"{tp} / {tp + fn}")
    c[1].metric("Rappel", f"{tp / max(tp + fn, 1):.0%}")
    c[2].metric("Précision des refus", f"{tp / max(tp + fp, 1):.0%}")
    c[3].metric("Bons payeurs refusés à tort", f"{fp}", help=f"sur {int((y == 0).sum())} bons payeurs")
    c[4].metric("Coût par demande", f"{(ratio * fn + fp) / len(y):.3f}")
    grid = np.linspace(0.01, 0.99, 99)
    costs = [(ratio * ((pr < t) & (y == 1)).sum() + ((pr >= t) & (y == 0)).sum()) / len(y) for t in grid]
    g1, g2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(grid, costs, color=C_GOOD, lw=2)
    ax.plot(thr, (ratio * fn + fp) / len(y), "o", ms=11, color=C_BAD, markeredgecolor="white", markeredgewidth=2)
    ax.axvline(grid[int(np.argmin(costs))], color="#888", ls="--", lw=1)
    ax.set_xlabel("Seuil de décision"); ax.set_ylabel("Coût moyen par demande")
    ax.set_title(f"Coût selon le seuil (minimum à {grid[int(np.argmin(costs))]:.2f} pour ce rapport de coûts)", fontsize=9)
    ax.grid(alpha=0.25); g1.pyplot(fig, clear_figure=True)
    prec, rec, _ = precision_recall_curve(y, pr)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(rec, prec, color=C_GOOD, lw=2)
    ax.plot(tp / max(tp + fn, 1), tp / max(tp + fp, 1), "o", ms=11, color=C_BAD, markeredgecolor="white", markeredgewidth=2)
    ax.set_xlabel("Rappel (part des défauts refusés)"); ax.set_ylabel("Précision (part de refus justifiés)")
    ax.set_title(f"Courbe précision-rappel sur le test (PR-AUC = {average_precision_score(y, pr):.3f})", fontsize=9)
    ax.grid(alpha=0.25); g2.pyplot(fig, clear_figure=True)
    st.caption("Courbes tracées avec le modèle enregistré (graine 0) ; le rapport donne la PR-AUC moyenne "
               "sur 3 graines (0,904).")
    st.caption(f"Seuil retenu dans le projet : {bundle['threshold']:.3f}. Il minimise le coût sur le jeu de "
               "développement pour un rapport de coûts de 5 pour 1, puis est appliqué tel quel au test.")

# ------------------------------------------------------------------------------------------
with tab_cmp:
    st.markdown("**Question du projet :** ajouter des défauts synthétiques à l'entraînement aide-t-il à mieux détecter "
                "les vrais ? Toutes les méthodes sont comparées sur les mêmes 15 découpages, puis sur un jeu de test "
                "tenu à l'écart jusqu'à la fin.")
    st.subheader("Validation croisée")
    best = load_csv(str(TAB / "cv_best.csv"))
    clf = st.radio("Classifieur", list(best["classifieur"].unique())[::-1], horizontal=True)
    cols = ["config", "PR-AUC", "PR-AUC (écart-type)", "ROC-AUC", "KS", "gain", "p-valeur (Holm)"]
    st.dataframe(best[best["classifieur"] == clf][cols].round(3), hide_index=True, width="stretch")
    st.caption("« gain » : écart de PR-AUC avec l'absence de rééquilibrage, à la meilleure dose de la méthode. "
               "Une p-valeur ajustée inférieure à 0,05 indiquerait un écart statistiquement significatif.")
    st.image(str(FIG / "cv_gains.png"))
    st.image(str(FIG / "cv_dose.png"))
    st.subheader("Fort déséquilibre : étude de rareté")
    st.image(str(FIG / "rarete_dose1.png"))
    st.subheader("Évaluation finale sur le jeu de test")
    res = load_csv(str(TAB / "test_results.csv"))
    st.dataframe(res[["classifieur", "config", "PR-AUC validation croisée", "PR-AUC test", "IC bas", "IC haut",
                      "différence vs Aucune", "diff. IC bas", "diff. IC haut"]].round(3),
                 hide_index=True, width="stretch")
    st.image(str(FIG / "test_pr_auc.png"))

# ------------------------------------------------------------------------------------------
with tab_syn:
    st.markdown("Les générateurs apprennent sur les vrais défauts d'entraînement, puis en inventent de nouveaux. "
                "Un bon défaut synthétique **ressemble** aux vrais sans en être une **copie**.")
    q = load_csv(str(TAB / "synthetic_quality.csv"), index_col=0)
    st.dataframe(q.round(3), width="stretch")
    st.caption("La première ligne donne les valeurs obtenues par de vrais défauts mis de côté : c'est la référence. "
               "Détectabilité idéale : 0,5. Rapport DCR idéal : 1 (proche de 0 : copies de clients réels).")
    samples = load_csv(str(APP_DATA / "synthetic_samples.csv"))
    c1, c2 = st.columns([1, 3])
    with c1:
        method = st.selectbox("Méthode", [m for m in samples["méthode"].unique() if m != "Vrais défauts"])
        var = st.selectbox("Variable", RAW_COLS, index=RAW_COLS.index("DEBTINC"), format_func=lambda v: f"{v} — {LABELS[v]}")
    with c2:
        real = samples[samples["méthode"] == "Vrais défauts"][var]
        fake = samples[samples["méthode"] == method][var]
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        if var in CAT_COLS:
            cats = CATEGORIES[var]
            p = real.fillna("Inconnu").value_counts(normalize=True).reindex(cats).fillna(0)
            f = fake.fillna("Inconnu").value_counts(normalize=True).reindex(cats).fillna(0)
            x = np.arange(len(cats))
            ax.bar(x - 0.2, 100 * p, 0.4, color="#bdbdbd", label="Vrais défauts")
            ax.bar(x + 0.2, 100 * f, 0.4, color=C_BAD, label=method)
            ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("% des défauts")
        else:
            lo, hi = real.quantile(0.005), real.quantile(0.995)
            bins = np.arange(lo - 0.5, hi + 1.5) if real.nunique() < 20 else np.linspace(lo, hi, 31)
            ax.hist(real.dropna().clip(bins[0], bins[-1]), bins=bins, density=True, color="#bdbdbd", label="Vrais défauts")
            ax.hist(fake.dropna().clip(bins[0], bins[-1]), bins=bins, density=True, histtype="step", lw=2,
                    color=C_BAD, label=method)
            ax.set_yticks([]); ax.set_xlabel(LABELS[var])
        ax.legend(frameon=False)
        st.pyplot(fig, clear_figure=True)
        st.caption(f"Valeurs manquantes — vrais défauts : {real.isna().mean():.0%} ; {method} : {fake.isna().mean():.0%}")
    st.image(str(FIG / "qualite_correlations.png"))
    st.image(str(FIG / "ddpm_tuning.png"), caption="Réglage de TabDDPM : s'entraîner plus longtemps améliore la "
             "fidélité, puis le modèle se rapproche de ses données d'entraînement.")

# ------------------------------------------------------------------------------------------
with tab_csv:
    st.markdown("Déposez un fichier CSV de demandes au format du jeu de données d'origine (colonnes "
                + ", ".join(f"`{c}`" for c in RAW_COLS) + "). Les cases vides sont acceptées.")
    up = st.file_uploader("Fichier CSV", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        missing = [c for c in RAW_COLS if c not in raw.columns]
        if missing:
            st.error(f"Colonnes manquantes : {', '.join(missing)}")
        else:
            out = raw.copy()
            out["proba_defaut"] = score(raw)
            out["decision"] = np.where(out["proba_defaut"] >= bundle["threshold"], "Refus", "Accord")
            st.success(f"{len(out)} demandes analysées, {int((out['decision'] == 'Refus').sum())} refus proposés.")
            st.dataframe(out.sort_values("proba_defaut", ascending=False).head(50), width="stretch")
            st.download_button("Télécharger les résultats", out.to_csv(index=False).encode("utf-8"),
                               "demandes_scorees.csv", "text/csv")
    else:
        st.download_button("Télécharger un fichier d'exemple",
                           (APP_DATA / "exemple_a_scorer.csv").read_bytes(), "exemple_a_scorer.csv", "text/csv")

# ------------------------------------------------------------------------------------------
with tab_about:
    st.markdown((ROOT / "app" / "a_propos.md").read_text(encoding="utf-8"))
