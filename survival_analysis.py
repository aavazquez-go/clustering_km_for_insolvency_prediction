#!/usr/bin/env python3
"""
Análisis de Supervivencia No Paramétrico - Kaplan-Meier
=======================================================
Estudia la supervivencia empresarial a partir de los datasets
de entrenamiento y prueba usando el estimador de Kaplan-Meier.

Los datos están en formato counting-process (múltiples filas por CIF).
Se colapsan a 1 fila por empresa usando la primera observación cronológica
como referencia basal.

Genera:
  - plots/   : gráficos de las funciones de supervivencia
  - logs/    : archivos con estadísticas detalladas
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

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
TRAIN_PATH = '/media/datos/WORKSPACE/DOCTORADO/datasets/train_set.csv'
TEST_PATH = '/media/datos/WORKSPACE/DOCTORADO/datasets/test_set.csv'
TRAIN_COLLAPSED = 'datasets/train_collapsed.csv'
TEST_COLLAPSED = 'datasets/test_collapsed.csv'
COMBINED_COLLAPSED = 'datasets/all_collapsed.csv'
PLOTS_DIR = 'plots'
LOGS_DIR = 'logs'
FIGURE_SIZE = (14, 8)

# Las variables categóricas ya vienen codificadas como one-hot en el dataset
CATEGORY_CODE_MAP = {
    'N3_Sector': {
        'col': 'N3',
        'code_map': {0: 'Agriculture', 1: 'Construction', 2: 'Industry', 3: 'Services'},
        'labels': ['Agricultura', 'Construccion', 'Industria', 'Servicios'],
    },
    'N4_LegalForm': {
        'col': 'N4',
        'code_map': {0: 'Cooperative', 1: 'Limited Company'},
        'labels': ['Cooperativa', 'S.L.'],
    },
    'N5': {
        'col': 'N5',
        'code_map': {0: 'No', 1: 'Yes'},
        'labels': ['No', 'Si'],
    },
    'N8_Size': {
        'col': 'N8',
        'code_map': {0: 'Microenterprise', 1: 'Small company', 2: 'Medium enterprise'},
        'labels': ['Microempresa', 'Pequeña', 'Mediana'],
    },
    'N9': {
        'col': 'N9',
        'code_map': {0: 'No', 1: 'Yes'},
        'labels': ['No', 'Si'],
    },
    'N10_AuditOpinion': {
        'col': 'N10',
        'code_map': {0: 'Favorable', 1: 'Qualified', 2: 'Unfavorable'},
        'labels': ['Favorable', 'Con salvedades', 'Desfavorable'],
    },
    'N11': {
        'col': 'N11',
        'code_map': {0: 'Female', 1: 'Male'},
        'labels': ['Femenino', 'Masculino'],
    },
    'N15': {
        'col': 'N15',
        'code_map': {0: 'No', 1: 'Yes'},
        'labels': ['No', 'Si'],
    },
}

# Construir CATEGORICAL_GROUPS dinámicamente a partir de CATEGORY_CODE_MAP
CATEGORICAL_GROUPS = {}
for group_name, cfg in CATEGORY_CODE_MAP.items():
    col = cfg['col']
    code_map = cfg['code_map']
    labels = cfg['labels']
    prefix = f'{col}_'
    columns = [f'{prefix}{code_map[k]}' for k in sorted(code_map.keys())]
    CATEGORICAL_GROUPS[group_name] = {'columns': columns, 'labels': labels}

COVARIATE_COLS = (
    ['N1', 'N2']
    + list({c for g in CATEGORICAL_GROUPS.values() for c in g['columns']})
    + ['N6', 'N7', 'N12', 'N13', 'N14']
    + [f'F{i}' for i in range(16, 26)]
    + [f'F{i}' for i in range(27, 32)]
    + [f'F{i}' for i in range(33, 44)]
)
# Remove duplicates preserving order
_COV_SET = set()
COVARIATE_COLS = [c for c in COVARIATE_COLS if not (c in _COV_SET or _COV_SET.add(c))]

NUMERICAL_VARS = (
    ['N1', 'N2', 'N6', 'N7', 'N12', 'N13', 'N14']
    + [f'F{i}' for i in range(16, 26)]
    + [f'F{i}' for i in range(27, 32)]
    + [f'F{i}' for i in range(33, 44)]
)


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


def load_and_prepare():
    """
    Carga los datasets colapsados (1 fila/empresa) si existen,
    o los genera desde el formato counting-process y los guarda.
    También genera el dataset combinado (train+test) si no existe.
    """
    if os.path.exists(TRAIN_COLLAPSED) and os.path.exists(TEST_COLLAPSED):
        train = pd.read_csv(TRAIN_COLLAPSED)
        test = pd.read_csv(TEST_COLLAPSED)
    else:
        raw_train = pd.read_csv(TRAIN_PATH)
        raw_test = pd.read_csv(TEST_PATH)

        def _collapse(df):
            df = df.sort_values(['CIF', 'Start']).reset_index(drop=True)
            cov_cols = [c for c in df.columns if c not in ('CIF', 'Start', 'Stop', 'Event')]
            baseline = df.groupby('CIF')[cov_cols].nth(0).reset_index()
            first_start = df.groupby('CIF')['Start'].min()
            last_stop = df.groupby('CIF')['Stop'].max()
            has_event = df.groupby('CIF')['Event'].max()
            time = last_stop - first_start
            event_stop = df[df['Event'] == 1].groupby('CIF')['Stop'].max()
            cifs_con_event = event_stop.index.intersection(time.index)
            time[cifs_con_event] = event_stop[cifs_con_event] - first_start[cifs_con_event]
            result = baseline.copy()
            result['Start'] = 0
            result['Stop'] = time.values
            result['Event'] = has_event.values.astype(int)
            return result

        train = _collapse(raw_train)
        test = _collapse(raw_test)
        train.to_csv(TRAIN_COLLAPSED, index=False)
        test.to_csv(TEST_COLLAPSED, index=False)

    if not os.path.exists(COMBINED_COLLAPSED):
        combined = pd.concat([
            train.assign(dataset='TRAIN'),
            test.assign(dataset='TEST'),
        ], ignore_index=True)
        combined.to_csv(COMBINED_COLLAPSED, index=False)

    return train, test, train.shape[0], test.shape[0]


def resolve_category(df, columns, labels):
    """Convierte columnas one-hot en una Serie categórica."""
    s = pd.Series(index=df.index, dtype=str)
    for col, lbl in zip(columns, labels):
        s[df[col] == 1] = lbl
    s.fillna('Desconocido', inplace=True)
    return s


def fit_km_safe(durations, events, entry, label, min_subjects=5):
    """Ajusta Kaplan-Meier con manejo de errores."""
    if len(durations) < min_subjects or events.sum() < 1:
        return None
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations, event_observed=events,
                entry=entry, label=label)
        return kmf
    except Exception:
        return None


# ============================================================
# LOGGING DE RESULTADOS
# ============================================================

def log_header(logger, title):
    logger.info('')
    logger.info('=' * 80)
    logger.info(f'  {title}')
    logger.info('=' * 80)


def log_base_results(logger, kmf, label, n_subjects=None):
    """Vuelca estadísticas de un único ajuste KM."""
    if n_subjects is None:
        try:
            n_subjects = int(kmf.event_table['at_risk'].iloc[0])
        except Exception:
            n_subjects = 0

    n_events = int(kmf.event_table['observed'].sum()) if kmf.event_table is not None else 0
    n_censored = n_subjects - n_events

    logger.info(f'  Grupo: {label}')
    logger.info(f'    Sujetos (empresas):     {n_subjects}')
    logger.info(f'    Eventos:                 {n_events}')
    logger.info(f'    Censurados:              {n_censored}')

    try:
        med = kmf.median_survival_time_
        logger.info(f'    Mediana supervivencia:   {med:.2f}')
    except Exception:
        logger.info(f'    Mediana supervivencia:   No alcanzada')

    # survival_function_at_times(t) returns a Series (index=t, value=surv) in lifelines 0.29
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

    logger.info(f'    {"─" * 50}')


def log_logrank(logger, durations, events, groups):
    """Test de log-rank multi-grupo."""
    present = groups.dropna().unique()
    if len(present) < 2:
        return
    mask = groups.isin(present)
    try:
        result = multivariate_logrank_test(
            durations[mask], groups[mask], events[mask]
        )
        logger.info(f'')
        logger.info(f'  Test de Log-Rank (multi-grupo)')
        logger.info(f'    Estadístico:  {result.test_statistic:.4f}')
        logger.info(f'    p-valor:      {result.p_value:.6e}')
        logger.info(f'    gl:           {result.degrees_of_freedom}')
    except Exception as e:
        logger.info(f'  Log-Rank no disponible: {e}')


# ============================================================
# PLOTS
# ============================================================

def plot_base(kmf, title, filename):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2, color='#2166ac')

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Umbral de riesgo (0.5)')

    sf = kmf.survival_function_
    col = sf.columns[0]
    mask = sf[col] < 0.5
    if mask.any():
        idx = mask.values.argmax()
        time = sf.index[idx]
        ax.axvline(x=time, color='#2166ac', linestyle='--', linewidth=1.5)
        ax.text(time, 0.95, f'{time:.0f}', color='#2166ac', fontsize=11,
                ha='center', va='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#2166ac', alpha=0.8))

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Tiempo', fontsize=13)
    ax.set_ylabel('Probabilidad de Supervivencia', fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_stratified(fitters, title, filename, p_value=None):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    n = len(fitters)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))

    for (name, kmf), color in zip(fitters.items(), colors):
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
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8))

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Umbral de riesgo (0.5)')

    if p_value is not None:
        ax.text(0.98, 0.05, f'Log-rank p = {p_value:.2e}',
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                ha='right', va='bottom')

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Tiempo', fontsize=13)
    ax.set_ylabel('Probabilidad de Supervivencia', fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# ANÁLISIS PRINCIPALES
# ============================================================

def analyze_base(datasets, logger):
    log_header(logger, 'FASE 1 — ANÁLISIS BASE (FUNCIÓN DE SUPERVIVENCIA GLOBAL)')

    for dset_name, df in datasets:
        logger.info(f'\n  --- {dset_name} ---\n')
        kmf = fit_km_safe(df['Stop'], df['Event'], df['Start'], 'Global')
        if kmf is None:
            logger.info('  (No fue posible ajustar el modelo global)')
            continue
        log_base_results(logger, kmf, 'Global', n_subjects=len(df))
        plot_base(kmf,
                  f'Función de Supervivencia Global — {dset_name}',
                  f'base_km_{dset_name.lower()}.png')


def analyze_categorical(datasets, logger):
    log_header(logger, 'FASE 2 — ANÁLISIS POR VARIABLES CATEGÓRICAS')

    for var_name, cfg in CATEGORICAL_GROUPS.items():
        log_header(logger, f'Variable: {var_name}')

        for dset_name, df in datasets:
            categories = resolve_category(df, cfg['columns'], cfg['labels'])
            present_labels = [l for l in cfg['labels'] if categories.eq(l).any()]

            fitters = {}
            for label in present_labels:
                mask = categories == label
                kmf = fit_km_safe(
                    df.loc[mask, 'Stop'],
                    df.loc[mask, 'Event'],
                    df.loc[mask, 'Start'],
                    label,
                    min_subjects=5,
                )
                if kmf is not None:
                    fitters[label] = kmf

            if not fitters:
                logger.info(f'\n  --- {var_name} — {dset_name} ---')
                logger.info('  (Ningún grupo válido para análisis)')
                continue

            logger.info(f'\n  --- {var_name} — {dset_name} ---\n')
            for label, kmf in fitters.items():
                n_subj = int((categories == label).sum())
                log_base_results(logger, kmf, label, n_subjects=n_subj)

            log_logrank(logger, df['Stop'], df['Event'], categories)

            pv = None
            if len(fitters) >= 2:
                mask = categories.isin(present_labels)
                try:
                    result = multivariate_logrank_test(
                        df.loc[mask, 'Stop'], categories[mask], df.loc[mask, 'Event'])
                    pv = result.p_value
                except Exception:
                    pass

            plot_stratified(fitters,
                            f'Supervivencia por {var_name} — {dset_name}',
                            f'{var_name}_{dset_name.lower()}.png',
                            p_value=pv)


def analyze_numerical(datasets, logger, split_mode='median'):
    split_label = 'MEDIANA' if split_mode == 'median' else 'CUARTILES'
    log_header(logger, f'FASE 3 — ANÁLISIS POR VARIABLES NUMÉRICAS ({split_label})')

    ref = datasets[0][1]

    split_dict = {}
    for var in NUMERICAL_VARS:
        if var not in ref.columns:
            continue
        vals = ref[var].dropna()
        if len(vals) < 100:
            continue
        q = vals.quantile([0.25, 0.5, 0.75]).values
        if split_mode == 'median':
            split_dict[var] = [q[1]]
        elif q[0] != q[2]:
            split_dict[var] = q

    fmt_label = 'mediana' if split_mode == 'median' else 'cuartiles'

    for var in NUMERICAL_VARS:
        if var not in ref.columns or var not in split_dict:
            continue

        log_header(logger, f'Variable numérica: {var}')
        splits = split_dict[var]
        bins_raw = [-np.inf] + list(splits) + [np.inf]
        bins_uniq = sorted(set(bins_raw))
        n_bins = len(bins_uniq) - 1
        if n_bins < 2:
            logger.info(f'  Variable {var}: {n_bins} grupo(s) tras eliminar duplicados, se omite.\n')
            continue
        def _fmt_val(v):
            return f'{v:.6f}' if np.isfinite(v) else ''
        bin_labels = []
        for i in range(n_bins):
            lo = bins_uniq[i]
            hi = bins_uniq[i + 1]
            lo_str = _fmt_val(lo)
            hi_str = _fmt_val(hi)
            if not lo_str:
                label = f'≤{hi_str}'
            elif not hi_str:
                label = f'>{lo_str}'
            else:
                label = f'({lo_str}, {hi_str}]'
            bin_labels.append(label)

        for dset_name, df in datasets:
            groups = pd.cut(df[var], bins=bins_uniq, labels=bin_labels, include_lowest=True)
            present_labels = [l for l in bin_labels if groups.eq(l).any()]

            fitters = {}
            for label in present_labels:
                mask = groups == label
                kmf = fit_km_safe(
                    df.loc[mask, 'Stop'],
                    df.loc[mask, 'Event'],
                    df.loc[mask, 'Start'],
                    label,
                    min_subjects=5,
                )
                if kmf is not None:
                    fitters[label] = kmf

            if not fitters:
                logger.info(f'\n  --- {var} ({fmt_label}) — {dset_name} ---')
                logger.info('  (Ningún grupo válido para análisis)')
                continue

            logger.info(f'\n  --- {var} ({fmt_label}) — {dset_name} ---\n')
            for label, kmf in fitters.items():
                n_subj = int((groups == label).sum())
                log_base_results(logger, kmf, label, n_subjects=n_subj)

            log_logrank(logger, df['Stop'], df['Event'], groups)

            pv = None
            if len(fitters) >= 2:
                mask = groups.isin(present_labels)
                try:
                    result = multivariate_logrank_test(
                        df.loc[mask, 'Stop'], groups[mask], df.loc[mask, 'Event'])
                    pv = result.p_value
                except Exception:
                    pass

            plot_stratified(fitters,
                            f'Supervivencia por {var} ({fmt_label}) — {dset_name}',
                            f'{var}_{dset_name.lower()}.png',
                            p_value=pv)


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de Supervivencia Kaplan-Meier')
    parser.add_argument('--mode', choices=['train', 'test', 'both', 'combined'], default='both',
                        help='Modo de análisis: train, test, both (default), combined')
    parser.add_argument('--split', choices=['median', 'quartiles'], default='median',
                        help='Estrategia de división para variables numéricas (default: median)')
    args = parser.parse_args()

    ensure_dir(PLOTS_DIR)
    ensure_dir(LOGS_DIR)

    logger = setup_logger('survival', os.path.join(LOGS_DIR, 'survival_analysis.log'))

    logger.info('=' * 80)
    logger.info('  ANALISIS DE SUPERVIVENCIA — KAPLAN-MEIER')
    logger.info('  Estimación no paramétrica de la función de supervivencia')
    logger.info(f'  Modo: {args.mode}  |  Split numérico: {args.split}')
    logger.info('=' * 80)

    logger.info('\n  Cargando y preparando datos...')
    train, test, n_raw_train, n_raw_test = load_and_prepare()
    logger.info(f'  Train: {n_raw_train} empresas')
    logger.info(f'  Test:  {n_raw_test} empresas')

    if args.mode == 'combined':
        all_data = pd.concat([
            train.assign(dataset='TRAIN'),
            test.assign(dataset='TEST'),
        ], ignore_index=True)
        datasets = [('COMBINADO', all_data)]
    elif args.mode == 'train':
        datasets = [('TRAIN', train)]
    elif args.mode == 'test':
        datasets = [('TEST', test)]
    else:
        datasets = [('TRAIN', train), ('TEST', test)]

    for dset_name, df in datasets:
        n_events = int(df['Event'].sum())
        logger.info(f'  {dset_name}: {len(df)} empresas, {n_events} eventos ({n_events/len(df)*100:.2f}%)')

    analyze_base(datasets, logger)
    analyze_categorical(datasets, logger)
    analyze_numerical(datasets, logger, split_mode=args.split)

    logger.info('')
    logger.info('=' * 80)
    logger.info('  ANALISIS COMPLETADO')
    logger.info(f'  Graficos  → {PLOTS_DIR}/')
    logger.info(f'  Logs      → {LOGS_DIR}/')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
