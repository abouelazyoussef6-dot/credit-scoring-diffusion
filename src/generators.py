"""Modèles génératifs profonds : création de clients défaillants synthétiques.

Principe, identique pour les trois générateurs :
1. on prend UNIQUEMENT les vrais défauts du jeu d'entraînement (jamais la validation ni
   le test) ;
2. le générateur apprend à quoi ressemble une demande de prêt qui finira en défaut ;
3. il produit de nouvelles demandes, au même format que les vraies (valeurs manquantes
   comprises), qu'on ajoutera aux données d'entraînement.

Générateurs :
- TVAE    : autoencodeur variationnel tabulaire (bibliothèque ctgan du projet SDV) ;
- CTGAN   : réseau antagoniste génératif conditionnel tabulaire (même bibliothèque) ;
- TabDDPM : modèle de diffusion tabulaire (bibliothèque synthcity, plugin « ddpm »).

Valeurs manquantes. Les générateurs ne les acceptent pas, mais leur présence est un signal
fort du défaut : il faut la reproduire. Chaque variable numérique incomplète est donc
décrite par deux colonnes dans la table d'apprentissage : sa valeur, et un indicateur
« manquant » (catégoriel). Quand la valeur manque, on la remplace par une valeur tirée au
hasard parmi les valeurs observées (ce qui ne déforme pas la distribution). Après
génération, la valeur est effacée partout où l'indicateur généré vaut « manquant ».

Deux précautions pour TabDDPM, justifiées par src.ddpm_tuning :
- déquantification : un bruit infime départage les valeurs répétées (nombre d'incidents
  égal à 0 pour la plupart des clients), sinon la normalisation par quantiles envoie les
  valeurs égales au minimum vers un point extrême et le modèle génère des valeurs collées
  aux bornes ;
- durée d'entraînement choisie pour être fidèle sans MÉMORISER les vrais clients.
"""
import json
import os
import warnings

os.environ.setdefault("TQDM_DISABLE", "1")  # masque les barres de progression de synthcity
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import CAT_COLS, INT_COLS, MISSING_LABEL, NUM_COLS, RAW_COLS, SYNTH_DIR, TAB_DIR
from src.features import CATEGORIES, IND_COLS

GENERATORS = ["TVAE", "CTGAN", "TabDDPM"]
SDV_EPOCHS = 300
DDPM_FILE = TAB_DIR / "ddpm_config.json"       # écrit par src.ddpm_tuning
DDPM_DEFAULT = {"n_iter": 5000, "lr": 0.003, "num_timesteps": 100}
FLAG_COLS = [f"{c}_manquant" for c in IND_COLS]
DISCRETE_COLS = CAT_COLS + FLAG_COLS


def ddpm_config() -> dict:
    cfg = dict(DDPM_DEFAULT)
    if DDPM_FILE.exists():
        cfg.update(json.loads(DDPM_FILE.read_text(encoding="utf-8")))
    return cfg


def to_table(X: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Table d'apprentissage des générateurs : sans valeur manquante, avec indicateurs."""
    rng = np.random.default_rng(seed)
    T = pd.DataFrame(index=X.index)
    for c in NUM_COLS:
        v = X[c].astype(float).copy()
        obs = v.dropna().to_numpy()
        if v.isna().any():
            v[v.isna()] = rng.choice(obs, size=int(v.isna().sum())) if len(obs) else 0.0
        T[c] = v
    for c in CAT_COLS:
        T[c] = X[c].where(X[c].isin(CATEGORIES[c][:-1]), MISSING_LABEL).astype(str)
    for c in IND_COLS:
        T[f"{c}_manquant"] = np.where(X[c].isna(), "oui", "non")
    return T.reset_index(drop=True)


def from_table(synth: pd.DataFrame, real_table: pd.DataFrame) -> pd.DataFrame:
    """Ramène une table générée au format des vraies demandes : bornes observées, valeurs
    entières arrondies, valeurs manquantes rétablies."""
    out = pd.DataFrame(index=synth.index)
    for c in NUM_COLS:
        v = pd.to_numeric(synth[c], errors="coerce").clip(real_table[c].min(), real_table[c].max())
        if c in INT_COLS:
            v = v.round()
        if c == "LOAN":
            v = (v / 100).round() * 100            # les prêts sont accordés par tranches de 100
        out[c] = v
    for c in CAT_COLS:
        s = synth[c].astype(str)
        out[c] = s.where(s.isin(CATEGORIES[c][:-1]))        # « Inconnu » redevient manquant
    for c in IND_COLS:
        out.loc[synth[f"{c}_manquant"].astype(str) == "oui", c] = np.nan
    return out.dropna(subset=["LOAN"])[RAW_COLS].reset_index(drop=True)


def dequantize(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Ajoute aux colonnes numériques un bruit uniforme plus petit que l'écart entre deux valeurs voisines."""
    rng = np.random.default_rng(seed)
    out = table.copy()
    for col in NUM_COLS:
        values = np.unique(table[col])
        step = float(np.diff(values).min()) if len(values) > 1 else 1.0
        out[col] = table[col] + rng.uniform(-step / 2, step / 2, len(table))
    return out


def fit_generate(kind: str, defaults: pd.DataFrame, n: int, seed: int,
                 ddpm: dict | None = None) -> pd.DataFrame:
    """Entraîne le générateur `kind` sur les vrais défauts et en génère `n` nouveaux."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    table = to_table(defaults, seed)

    if kind in ("TVAE", "CTGAN"):
        from ctgan import CTGAN, TVAE

        batch = max(10, min(500, len(table) // 10 * 10))   # multiple de 10 exigé par CTGAN
        cls = TVAE if kind == "TVAE" else CTGAN
        model = cls(epochs=SDV_EPOCHS, batch_size=batch, verbose=False)
        model.fit(table, discrete_columns=DISCRETE_COLS)
        synth = model.sample(int(n * 1.02) + 5)

    elif kind == "TabDDPM":
        from synthcity.plugins import Plugins

        cfg = ddpm_config() if ddpm is None else ddpm
        model = Plugins().get(
            "ddpm", n_iter=cfg["n_iter"], lr=cfg["lr"], num_timesteps=cfg["num_timesteps"],
            batch_size=min(1024, len(table)), random_state=seed,
        )
        model.fit(dequantize(table, seed))
        synth = model.generate(count=int(n * 1.02) + 5).dataframe()
    else:
        raise ValueError(kind)

    return from_table(synth, table).iloc[:n]


def tag(kind: str) -> str:
    """Identifie les réglages dans le nom du fichier en cache."""
    if kind == "TabDDPM":
        c = ddpm_config()
        return f"it{c['n_iter']}_lr{c['lr']}_T{c['num_timesteps']}"
    return f"ep{SDV_EPOCHS}"


def get_synthetic(kind: str, key: str, defaults: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Défauts synthétiques pour un découpage donné (`key`), avec cache sur disque."""
    path = SYNTH_DIR / f"{kind}_{key}_{tag(kind)}.csv"
    if path.exists():
        synth = pd.read_csv(path)
        if len(synth) >= n:
            return synth.iloc[:n]
    synth = fit_generate(kind, defaults, n, seed)
    synth.to_csv(path, index=False)
    return synth
