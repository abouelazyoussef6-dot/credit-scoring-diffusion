"""Étude 2 — défauts (faillites) synthétiques : mêmes générateurs que l'étude 1.

Seules les vraies faillites d'ENTRAÎNEMENT du pli servent à entraîner un générateur.
Adaptations aux ratios financiers :
- les générateurs travaillent sur les ratios transformés par le logarithme signé (les
  valeurs extrêmes rendraient leur apprentissage instable), puis on revient à l'échelle
  d'origine et on borne aux valeurs observées ;
- un ratio manquant dans au moins 1 % des faillites d'entraînement est décrit par sa
  valeur (complétée au hasard parmi les valeurs observées) et un indicateur « oui/non » ;
  la valeur est effacée après génération là où l'indicateur vaut « oui ».
"""
import json
import os
import warnings

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.faillites.config import FEATURES, MISSING_FLAG_MIN, SYNTH, TAB
from src.faillites.features import slog, slog_inv

GENERATORS = ["TVAE", "CTGAN", "TabDDPM"]
SDV_EPOCHS = 300
DDPM_FILE = TAB / "ddpm_config.json"
DDPM_DEFAULT = {"n_iter": 5000, "lr": 0.003, "num_timesteps": 100, "scheduler": "linear"}


def ddpm_config() -> dict:
    cfg = dict(DDPM_DEFAULT)
    if DDPM_FILE.exists():
        cfg.update(json.loads(DDPM_FILE.read_text(encoding="utf-8")))
    return cfg


def flag_cols(defaults: pd.DataFrame) -> list[str]:
    return [c for c in FEATURES if defaults[c].isna().mean() >= MISSING_FLAG_MIN]


def to_table(X: pd.DataFrame, flags: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = pd.DataFrame(index=X.index)
    for c in FEATURES:
        v = slog(X[c].astype(float))
        obs = v.dropna().to_numpy()
        if v.isna().any():
            v[v.isna()] = rng.choice(obs, size=int(v.isna().sum())) if len(obs) else 0.0
        T[c] = v
    for c in flags:
        T[f"{c}_manquant"] = np.where(X[c].isna(), "oui", "non")
    return T.reset_index(drop=True)


def from_table(synth: pd.DataFrame, real: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=synth.index)
    for c in FEATURES:
        v = slog_inv(pd.to_numeric(synth[c], errors="coerce").astype(float))
        out[c] = v.clip(real[c].min(), real[c].max())
    for c in flags:
        out.loc[synth[f"{c}_manquant"].astype(str) == "oui", c] = np.nan
    return out.dropna(how="all").reset_index(drop=True)


def dequantize(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = table.copy()
    for col in FEATURES:
        values = np.unique(table[col])
        step = float(np.diff(values).min()) if len(values) > 1 else 1e-3
        out[col] = table[col] + rng.uniform(-step / 2, step / 2, len(table))
    return out


def fit_generate(kind: str, defaults: pd.DataFrame, n: int, seed: int, ddpm: dict | None = None) -> pd.DataFrame:
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    flags = flag_cols(defaults)
    table = to_table(defaults, flags, seed)
    discrete = [f"{c}_manquant" for c in flags]

    if kind in ("TVAE", "CTGAN"):
        from ctgan import CTGAN, TVAE
        batch = max(10, min(500, len(table) // 10 * 10))
        cls = TVAE if kind == "TVAE" else CTGAN
        model = cls(epochs=SDV_EPOCHS, batch_size=batch, verbose=False)
        model.fit(table, discrete_columns=discrete)
        synth = model.sample(int(n * 1.02) + 5)
    elif kind == "TabDDPM":
        # Modèle TabDDPM de synthcity appelé directement, avec notre propre encodage :
        # - ratios normalisés par quantiles à pleine résolution (la surcouche de la
        #   bibliothèque n'en garde qu'un pour 30 lignes, trop grossier pour ~300 faillites) ;
        # - indicateurs de valeur manquante codés en colonnes NUMÉRIQUES (±1 + bruit) : codés
        #   en catégories, la diffusion mixte gaussienne/multinomiale diverge en dimension 75
        #   (écart-type généré ≈ 270 au lieu de 1) ;
        # - calendrier de bruit linéaire (le calendrier cosinus diverge lui aussi).
        from sklearn.preprocessing import QuantileTransformer
        from synthcity.plugins.core.models.tabular_ddpm import TabDDPM
        cfg = ddpm_config() if ddpm is None else ddpm
        rng = np.random.default_rng(seed)
        qt = QuantileTransformer(n_quantiles=len(table), output_distribution="normal",
                                 random_state=seed).fit(table[FEATURES])
        fit_table = pd.DataFrame(qt.transform(table[FEATURES]), columns=FEATURES)
        for c in discrete:
            fit_table[c] = (table[c] == "oui").astype(float) * 2 - 1 + rng.uniform(-0.3, 0.3, len(table))
        model = TabDDPM(n_iter=cfg["n_iter"], lr=cfg["lr"], num_timesteps=cfg["num_timesteps"],
                        scheduler=cfg.get("scheduler", "linear"), batch_size=min(1024, len(table)))
        model.fit(fit_table)
        synth = model.generate(int(n * 1.02) + 5)
        synth[FEATURES] = qt.inverse_transform(synth[FEATURES].to_numpy(dtype=float))
        for c in discrete:
            synth[c] = np.where(synth[c].to_numpy(dtype=float) > 0, "oui", "non")
    else:
        raise ValueError(kind)
    return from_table(synth, defaults, flags).iloc[:n]


def tag(kind: str) -> str:
    if kind == "TabDDPM":
        c = ddpm_config()
        return f"it{c['n_iter']}_lr{c['lr']}_T{c['num_timesteps']}_{c.get('scheduler', 'linear')}_num"
    return f"ep{SDV_EPOCHS}"


def get_synthetic(kind: str, key: str, defaults: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    path = SYNTH / f"{kind}_{key}_{tag(kind)}.csv"
    if path.exists():
        synth = pd.read_csv(path)
        if len(synth) >= n:
            return synth.iloc[:n]
    synth = fit_generate(kind, defaults, n, seed)
    synth.to_csv(path, index=False)
    return synth
