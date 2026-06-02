#!/usr/bin/env python3
"""
Estratificación multivariante mediante descubrimiento de arquetipos financieros
==============================================================================
Integra aprendizaje no supervisado (K-Means) con análisis de supervivencia
no paramétrico (Kaplan-Meier) para identificar perfiles de riesgo
de insolvencia en PYMEs.

Fases:
  1. Clustering: K-Means sobre ratios financieros estandarizados.
     K óptimo mediante codo + silhouette.
  2. Validación: Kaplan-Meier estratificado por arquetipo + log-rank.
  3. Perfilado: interpretación de los arquetipos (medias por cluster).
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
COMBINED_COLLAPSED = 'datasets/all_collapsed.csv'
PLOTS_DIR = 'plots'
LOGS_DIR = 'logs'
FIGURE_SIZE = (14, 8)

RATIO_COLS = (
    [f'F{i}' for i in range(16, 26)]
    + [f'F{i}' for i in range(27, 32)]
    + [f'F{i}' for i in range(33, 44)]
)

CLUSTER_RANGE = range(2, 11)
RANDOM_STATE = 42
MAX_KMEANS_ITER = 300
N_INIT = 10
OUTLIER_PERCENTILE = 99.0


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def log_header(logger, title):
    logger.info('')
    logger.info('=' * 80)
    logger.info(f'  {title}')
    logger.info('=' * 80)


# ============================================================
# FASE 1: CLUSTERING
# ============================================================

def load_and_prepare():
    df = pd.read_csv(COMBINED_COLLAPSED)
    available = [c for c in RATIO_COLS if c in df.columns]
    missing = set(RATIO_COLS) - set(available)
    if missing:
        print(f'  AVISO: Columnas no encontradas: {sorted(missing)}')
    print(f'  Ratios financieros disponibles: {len(available)}')
    print(f'  Total empresas: {len(df):,}')
    print(f'  Eventos: {int(df["Event"].sum()):,} ({df["Event"].mean()*100:.1f}%)')
    return df, available


def remove_outliers(X, percentile=OUTLIER_PERCENTILE, logger=None):
    center = X.mean(axis=0)
    dists = np.linalg.norm(X - center, axis=1)
    threshold = np.percentile(dists, percentile)
    mask = dists <= threshold
    n_out = (~mask).sum()
    if logger:
        logger.info(f'  Outliers eliminados: {n_out} ({n_out/len(X)*100:.2f}%)')
        logger.info(f'  Muestra retenida: {mask.sum():,} empresas')
    else:
        print(f'  Outliers eliminados: {n_out} ({n_out/len(X)*100:.2f}%)')
    return mask


SAMPLE_FOR_K = 20000

def find_optimal_k(X, k_range=CLUSTER_RANGE, logger=None):
    inertias = []
    sil_scores = []

    sample_size = min(SAMPLE_FOR_K, len(X))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), sample_size, replace=False)
    Xs = X[idx]

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                    max_iter=MAX_KMEANS_ITER, n_init=N_INIT)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sil = silhouette_score(Xs, labels)
        sil_scores.append(sil)
        msg = f'  K={k:2d}  inercia={km.inertia_:>10.0f}  silhouette={sil:.4f}'
        if logger:
            logger.info(msg)
        else:
            print(msg)

    best_k = k_range[np.argmax(sil_scores)]
    if logger:
        logger.info(f'\n  -> K óptimo (silhouette): {best_k}')
    else:
        print(f'\n  -> K óptimo (silhouette): {best_k}')

    return best_k, inertias, sil_scores


def plot_elbow_silhouette(k_range, inertias, sil_scores, filename='elbow_silhouette.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    kr = list(k_range)

    ax1.plot(kr, inertias, marker='o', linewidth=2, color='#2166ac')
    ax1.set_title('Método del Codo (Inercia)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Número de clusters (K)', fontsize=12)
    ax1.set_ylabel('Inercia', fontsize=12)
    ax1.grid(True, alpha=0.25)
    ax1.set_xticks(kr)

    ax2.plot(kr, sil_scores, marker='s', linewidth=2, color='#b2182b')
    ax2.set_title('Coeficiente de Silueta', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Número de clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.grid(True, alpha=0.25)
    ax2.set_xticks(kr)

    best_k = kr[np.argmax(sil_scores)]
    ax2.axvline(x=best_k, color='green', linestyle='--', linewidth=1.5,
                label=f'K óptimo = {best_k}')
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def run_kmeans(X, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                max_iter=MAX_KMEANS_ITER, n_init=N_INIT)
    labels = km.fit_predict(X)
    return km, labels


def assign_outliers(X_full, X_filtered, km, filter_mask):
    full_labels = np.full(X_full.shape[0], -1, dtype=int)
    full_labels[filter_mask] = km.labels_
    outliers = ~filter_mask
    if outliers.any():
        dists = km.transform(X_full[outliers])
        full_labels[outliers] = dists.argmin(axis=1)
    return full_labels


def profile_archetypes(df, labels, ratio_cols, logger):
    df_prof = df[ratio_cols].copy()
    df_prof['Archetype'] = labels

    profile = df_prof.groupby('Archetype').agg(['mean', 'std', 'count'])
    profile.columns = [f'{col}_{stat}' for col, stat in profile.columns]
    profile = profile.reset_index()

    total = len(df_prof)
    logger.info(f'\n  {"Arquetipo":>10s}  {"n":>8s}  {"%":>7s}')
    logger.info(f'  {"-"*27}')
    count_col = [c for c in profile.columns if c.endswith('_count')][0]
    for _, row in profile.iterrows():
        arch = int(row['Archetype'])
        n = int(row[count_col])
        pct = n / total * 100
        logger.info(f'  {arch:>10d}  {n:>8,d}  {pct:>6.2f}%')

    profile_summary = []
    for arch in sorted(profile['Archetype'].unique()):
        mask = labels == arch
        n = int(mask.sum())
        if n < 5:
            continue
        means = df[ratio_cols].loc[mask].mean()
        arch_name = interpret_archetype(means)
        profile_summary.append({'Archetype': arch, 'Label': arch_name, 'n': n})

    logger.info('')
    for ps in profile_summary:
        logger.info(f'  Arquetipo {ps["Archetype"]}: {ps["Label"]}  (n={ps["n"]:,})')

    return profile, profile_summary


def interpret_archetype(means):
    top3 = means.abs().sort_values(ascending=False).head(3)
    high = top3[top3 > 0.3]
    low = top3[top3 < -0.3]
    parts = []
    for idx in high.index:
        parts.append(f'{idx}↑')
    for idx in low.index:
        parts.append(f'{idx}↓')
    return ' | '.join(parts[:3]) if parts else 'Perfil equilibrado'


def plot_archetype_profiles(profile, ratio_cols, filename='archetype_profiles.png'):
    archs = sorted(profile['Archetype'].unique())
    mean_cols = [f'{c}_mean' for c in ratio_cols]
    mean_cols = [c for c in mean_cols if c in profile.columns]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(ratio_cols))
    width = 0.8 / len(archs)
    colors = plt.cm.Set2(np.linspace(0, 1, len(archs)))

    for i, arch in enumerate(archs):
        row = profile[profile['Archetype'] == arch]
        vals = [row[c].values[0] for c in mean_cols]
        offset = (i - len(archs) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=f'Arquetipo {arch}', color=colors[i])

    ax.set_title('Perfil de Ratios Financieros por Arquetipo', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ratio', fontsize=12)
    ax.set_ylabel('Media (estandarizada)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ratio_cols, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.15, axis='y')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# FASE 2: SUPERVIVENCIA
# ============================================================

def fit_km_safe(durations, events, entry, label, min_subjects=5):
    if len(durations) < min_subjects or events.sum() < 1:
        return None
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations, event_observed=events,
                entry=entry, label=label)
        return kmf
    except Exception:
        return None


def log_base_results(logger, kmf, label, n_subjects=None):
    if n_subjects is None:
        try:
            n_subjects = int(kmf.event_table['at_risk'].iloc[0])
        except Exception:
            n_subjects = 0
    n_events = int(kmf.event_table['observed'].sum()) if kmf.event_table is not None else 0
    n_censored = n_subjects - n_events
    logger.info(f'  Arquetipo: {label}')
    logger.info(f'    Sujetos:     {n_subjects:>8,}')
    logger.info(f'    Eventos:     {n_events:>8,}')
    logger.info(f'    Censurados:  {n_censored:>8,}')
    try:
        med = kmf.median_survival_time_
        logger.info(f'    Mediana:     {med:.2f} años')
    except Exception:
        logger.info(f'    Mediana:     No alcanzada')
    for t in [1, 2, 3, 5, 10, 15, 20]:
        try:
            surv_s = kmf.survival_function_at_times(t)
            surv = surv_s.iloc[0] if hasattr(surv_s, 'iloc') else float(surv_s)
            ci = kmf.confidence_interval_survival_function_
            idx = ci.index.searchsorted(t)
            if idx < len(ci) and ci.shape[1] >= 2:
                lb = ci.iloc[idx, 0]
                ub = ci.iloc[idx, 1]
                logger.info(f'    S(t={t:<2d}) = {surv:.4f}  (IC 95%: {lb:.4f} - {ub:.4f})')
            else:
                logger.info(f'    S(t={t:<2d}) = {surv:.4f}')
        except Exception:
            pass
    logger.info(f'    {"─" * 55}')


def analyze_survival(df, labels, logger):
    log_header(logger, 'FASE 2 — ANÁLISIS DE SUPERVIVENCIA POR ARQUETIPO')

    unique_labels = sorted(set(labels))
    fitters = {}

    for lbl in unique_labels:
        mask = labels == lbl
        kmf = fit_km_safe(
            df.loc[mask, 'Stop'],
            df.loc[mask, 'Event'],
            df.loc[mask, 'Start'],
            f'Arquetipo {lbl}',
            min_subjects=5,
        )
        if kmf is not None:
            fitters[lbl] = kmf

    if not fitters:
        logger.info('  (Ningún grupo válido)')
        return None

    logger.info(f'\n  --- Curvas de supervivencia por arquetipo ---\n')
    for lbl, kmf in fitters.items():
        n_subj = int((labels == lbl).sum())
        log_base_results(logger, kmf, lbl, n_subjects=n_subj)

    present = list(fitters.keys())
    if len(present) >= 2:
        mask = np.isin(labels, present)
        try:
            result = multivariate_logrank_test(
                df.loc[mask, 'Stop'], labels[mask], df.loc[mask, 'Event']
            )
            logger.info(f'')
            logger.info(f'  Test de Log-Rank (multi-grupo)')
            logger.info(f'    Estadístico:  {result.test_statistic:.4f}')
            logger.info(f'    p-valor:      {result.p_value:.6e}')
            logger.info(f'    gl:           {result.degrees_of_freedom}')
            p_value = result.p_value
        except Exception as e:
            logger.info(f'  Log-Rank no disponible: {e}')
            p_value = None
    else:
        p_value = None

    plot_stratified_survival(fitters, p_value)
    return p_value


def plot_stratified_survival(fitters, p_value=None,
                             filename='km_archetypes.png'):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    n = len(fitters)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))

    for (lbl, kmf), color in zip(fitters.items(), colors):
        kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2, color=color)
        sf = kmf.survival_function_
        col = sf.columns[0]
        mask = sf[col] < 0.5
        if mask.any():
            idx = mask.values.argmax()
            time = sf.index[idx]
            ax.axvline(x=time, color=color, linestyle='--', linewidth=1.5)
            ax.text(time, 0.95, f'{time:.0f}', color=color, fontsize=10,
                    ha='center', va='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.8))

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5,
               label='Umbral de riesgo (0.5)')

    if p_value is not None:
        ax.text(0.98, 0.05, f'Log-rank p = {p_value:.2e}',
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                ha='right', va='bottom')

    ax.set_title('Supervivencia por Arquetipo Financiero',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Tiempo (años)', fontsize=13)
    ax.set_ylabel('Probabilidad de Supervivencia', fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Descubrimiento de Arquetipos Financieros + Supervivencia')
    parser.add_argument('--k', type=int, default=None,
                        help='Número de clusters (default: optimización automática)')
    parser.add_argument('--k-range', type=str, default='2,10',
                        help='Rango de K (default: 2,10)')
    parser.add_argument('--outlier-pct', type=float, default=OUTLIER_PERCENTILE,
                        help='Percentil para eliminación de outliers (default: 99)')
    parser.add_argument('--no-outlier-filter', action='store_true',
                        help='Deshabilitar eliminación de outliers')
    args = parser.parse_args()

    ensure_dir(PLOTS_DIR)
    ensure_dir(LOGS_DIR)

    logger = setup_logger('archetype',
                          os.path.join(LOGS_DIR, 'archetype_clustering.log'))

    logger.info('=' * 80)
    logger.info('  DESCUBRIMIENTO DE ARQUETIPOS FINANCIEROS')
    logger.info('  Clustering + Análisis de Supervivencia')
    logger.info('=' * 80)

    # ---- Carga ----
    logger.info('\n  Cargando datos...')
    df, ratio_cols = load_and_prepare()
    logger.info(f'  Ratios utilizados: {len(ratio_cols)}')
    logger.info(f'  Muestra: {len(df):,} empresas')

    # ---- Estandarización ----
    logger.info('\n  Estandarizando ratios...')
    scaler = StandardScaler()
    X_full = scaler.fit_transform(df[ratio_cols].values)

    # ---- Filtrado de outliers ----
    log_header(logger, 'FASE 1a — FILTRADO DE OUTLIERS')
    if args.no_outlier_filter:
        filter_mask = np.ones(len(X_full), dtype=bool)
        logger.info('  (Filtrado deshabilitado)')
    else:
        filter_mask = remove_outliers(X_full, args.outlier_pct, logger)
    X = X_full[filter_mask]

    # ---- K óptimo ----
    k_range_parts = args.k_range.split(',')
    k_range = range(int(k_range_parts[0]), int(k_range_parts[1]) + 1)

    if args.k is None:
        log_header(logger, 'FASE 1b — DETERMINACIÓN DE K ÓPTIMO')
        best_k, inertias, sil_scores = find_optimal_k(X, k_range, logger)
        plot_elbow_silhouette(k_range, inertias, sil_scores)
        logger.info(f'  Gráfico guardado: {PLOTS_DIR}/elbow_silhouette.png')
    else:
        best_k = args.k
        logger.info(f'\n  K fijado por usuario: {best_k}')

    # ---- K-Means ----
    log_header(logger, f'FASE 1c — K-MEANS (K={best_k})')
    km, _ = run_kmeans(X, best_k)
    logger.info(f'  Inercia final: {km.inertia_:.2f}')
    logger.info(f'  Tamaños de cluster: {np.bincount(km.labels_)}')

    # Asignar outliers al cluster más cercano
    labels = assign_outliers(X_full, X, km, filter_mask)
    logger.info(f'  Tamaños (con outliers asignados): {np.bincount(labels)}')

    # ---- Perfilado ----
    log_header(logger, 'FASE 1d — PERFIL DE ARQUETIPOS')
    profile, profile_summary = profile_archetypes(df, labels, ratio_cols, logger)
    plot_archetype_profiles(profile, ratio_cols)
    logger.info(f'  Gráfico guardado: {PLOTS_DIR}/archetype_profiles.png')

    # ---- Supervivencia ----
    p_value = analyze_survival(df, labels, logger)
    logger.info(f'\n  Gráfico guardado: {PLOTS_DIR}/km_archetypes.png')

    # ---- Resumen final ----
    logger.info('')
    logger.info('=' * 80)
    logger.info('  RESUMEN')
    logger.info('=' * 80)
    logger.info(f'  K óptimo:                 {best_k}')
    logger.info(f'  Arquetipos descubiertos:  {best_k}')
    logger.info(f'  Ratios utilizados:        {len(ratio_cols)}')
    for ps in profile_summary:
        logger.info(f'  Arquetipo {ps["Archetype"]}: {ps["Label"]} (n={ps["n"]:,})')
    if p_value is not None:
        logger.info(f'  Log-rank p-valor:         {p_value:.6e}')
    logger.info('')
    logger.info('  Archivos generados:')
    logger.info(f'    {PLOTS_DIR}/elbow_silhouette.png')
    logger.info(f'    {PLOTS_DIR}/archetype_profiles.png')
    logger.info(f'    {PLOTS_DIR}/km_archetypes.png')
    logger.info(f'    {LOGS_DIR}/archetype_clustering.log')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
