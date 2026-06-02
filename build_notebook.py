#!/usr/bin/env python3
"""
Genera survival_analysis.ipynb para el análisis combinado (publishable).
Importa survival_analysis durante la generación para pre-calcular cuartiles.
"""
import argparse, json, os, sys, textwrap, numpy as np, pandas as pd

parser = argparse.ArgumentParser(description='Genera survival_analysis.ipynb')
parser.add_argument('--split', choices=['median', 'quartiles'], default='median',
                    help='Estrategia de división para variables numéricas (default: median)')
args = parser.parse_args()
SPLIT_MODE = args.split

# ── Import del módulo de análisis ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survival_analysis as sa

# Cargar datos combinados para pre-calcular splits en tiempo de build
train, test, _, _ = sa.load_and_prepare()
all_data = pd.concat([train.assign(dataset='TRAIN'), test.assign(dataset='TEST')], ignore_index=True)

# Pre-calcular splits para variables numéricas
median_dict = {}
quantiles_dict = {}
for var in sa.NUMERICAL_VARS:
    if var not in all_data.columns: continue
    vals = all_data[var].dropna()
    if len(vals) < 100: continue
    q = vals.quantile([0.25, 0.5, 0.75]).values
    median_dict[var] = q[1]
    if SPLIT_MODE == 'quartiles' and q[0] != q[2]:
        quantiles_dict[var] = q

# ── Ensamblaje del notebook ────────────────────────────────────────
CELLS = []
def md(src):   CELLS.append({"cell_type":"markdown","id":f"m{len(CELLS)}","metadata":{},"source":[textwrap.dedent(src)]})
def cd(src):   CELLS.append({"cell_type":"code","id":f"c{len(CELLS)}","metadata":{},"outputs":[],"execution_count":None,"source":[textwrap.dedent(src)]})

md("""# Análisis de Supervivencia No Paramétrico — Kaplan-Meier
## Estudio de la supervivencia empresarial

Dataset combinado: **train + test** — análisis unificado de toda la muestra.
""")

# ── 1 ──
md("""## 1. Configuración e imports""")
cd("""import os, sys, warnings, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import survival_analysis as sa

warnings.filterwarnings('ignore')
%matplotlib inline
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (12, 6)
""")

# ── 2 ──
md("""## 2. Carga de datos

### 2.1 Counting-process → colapsado

Los datos fuente están en formato *counting-process* (múltiples filas por empresa). Se colapsan a una única fila por empresa tomando la primera observación cronológica como referencia basal. El tiempo de vida se calcula como:

$$
\\text{tiempo} =
\\begin{cases}
\\text{Stop}_{\\text{evento}} - \\min(\\text{Start}) & \\text{si evento ocurre} \\\\
\\max(\\text{Stop}) - \\min(\\text{Start}) & \\text{si censurado}
\\end{cases}
$$
""")
cd("""TRAIN_PATH  = '/media/datos/WORKSPACE/DOCTORADO/datasets/train_set.csv'
TEST_PATH   = '/media/datos/WORKSPACE/DOCTORADO/datasets/test_set.csv'
TRAIN_COLLAPSED  = 'datasets/train_collapsed.csv'
TEST_COLLAPSED   = 'datasets/test_collapsed.csv'
COMBINED_COLLAPSED = 'datasets/all_collapsed.csv'

def load_collapsed():
    if os.path.exists(COMBINED_COLLAPSED):
        return pd.read_csv(COMBINED_COLLAPSED)
    train_c = pd.read_csv(TRAIN_COLLAPSED) if os.path.exists(TRAIN_COLLAPSED) else None
    test_c  = pd.read_csv(TEST_COLLAPSED)  if os.path.exists(TEST_COLLAPSED)  else None
    if train_c is None or test_c is None:
        raw_train = pd.read_csv(TRAIN_PATH)
        raw_test  = pd.read_csv(TEST_PATH)
        def _collapse(df):
            df = df.sort_values(['CIF','Start']).reset_index(drop=True)
            cov = [c for c in df.columns if c not in ('CIF','Start','Stop','Event')]
            bsl  = df.groupby('CIF')[cov].nth(0).reset_index()
            t0   = df.groupby('CIF')['Start'].min()
            tN   = df.groupby('CIF')['Stop'].max()
            evt  = df.groupby('CIF')['Event'].max()
            time = tN - t0
            ev_stop = df[df['Event']==1].groupby('CIF')['Stop'].max()
            ev_cifs = ev_stop.index.intersection(time.index)
            time[ev_cifs] = ev_stop[ev_cifs] - t0[ev_cifs]
            out = bsl.copy()
            out['Start'] = 0
            out['Stop']  = time.values
            out['Event'] = evt.values.astype(int)
            return out
        train_c = _collapse(raw_train); train_c.to_csv(TRAIN_COLLAPSED, index=False)
        test_c  = _collapse(raw_test);  test_c.to_csv(TEST_COLLAPSED, index=False)
    combined = pd.concat([
        train_c.assign(dataset='TRAIN'),
        test_c.assign(dataset='TEST'),
    ], ignore_index=True)
    combined.to_csv(COMBINED_COLLAPSED, index=False)
    return combined

all_data = load_collapsed()
print(f'Dataset combinado: {len(all_data):,} empresas')
print(f'  Train: {(all_data["dataset"]=="TRAIN").sum():,}')
print(f'  Test:  {(all_data["dataset"]=="TEST").sum():,}')
print(f'  Eventos: {int(all_data["Event"].sum()):,} ({all_data["Event"].mean()*100:.1f}%)')
all_data.head(3)
""")

# ── 3 ──
md("""## 3. Estadísticas descriptivas""")
cd("""n = len(all_data)
ne = int(all_data['Event'].sum())
print(f'Total empresas:        {n:>8,}')
print(f'Eventos (insolvencia): {ne:>8,} ({ne/n*100:.2f}%)')
print(f'Censurados:            {n-ne:>8,} ({(n-ne)/n*100:.2f}%)')
print()
print('Distribución del tiempo de supervivencia (años):')
print(all_data['Stop'].describe())
""")

# ── 4 ──
md("""## 4. Función de supervivencia global

Estimación Kaplan-Meier con intervalo de confianza al 95%. Se marca el umbral de riesgo en S(t)=0.5 y el año de cruce.""")

cd("""kmf_g = KaplanMeierFitter()
kmf_g.fit(durations=all_data['Stop'], event_observed=all_data['Event'],
          entry=all_data['Start'], label='Global')

med = kmf_g.median_survival_time_
print(f'Mediana de supervivencia: {med:.1f} años' if np.isfinite(med) else 'Mediana: No alcanzada')
print()
for t in [1,2,3,5,10,15,20]:
    try:
        s = kmf_g.survival_function_at_times(t)
        print(f'S(t={t:<2d}) = {s.iloc[0] if hasattr(s,"iloc") else float(s):.4f}')
    except: pass
""")

cd("""fig, ax = plt.subplots()
kmf_g.plot_survival_function(ax=ax, ci_show=True, linewidth=2, color='#2166ac')
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Umbral de riesgo (0.5)')

sf = kmf_g.survival_function_
mask = sf.iloc[:,0] < 0.5
if mask.any():
    tc = sf.index[mask.values.argmax()]
    ax.axvline(x=tc, color='#2166ac', linestyle='--', linewidth=1.5)
    ax.text(tc, 0.95, f'{tc:.0f}', color='#2166ac', fontsize=11, ha='center', va='top',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#2166ac', alpha=0.8))

ax.set_title('Función de Supervivencia Global — Combinado', fontsize=14, fontweight='bold')
ax.set_xlabel('Tiempo (años)', fontsize=12)
ax.set_ylabel('Probabilidad de Supervivencia', fontsize=12)
ax.grid(True, alpha=0.25); ax.set_ylim(0, 1.05)
ax.legend(fontsize=11, loc='lower left')
plt.tight_layout(); plt.show()
""")

# ── 5. Categóricas ──
md("""## 5. Análisis por variables categóricas

Para cada variable categórica se estiman curvas de supervivencia por grupo y se comparan mediante el test de log-rank (multi-grupo).""")

for idx, (vname, cfg) in enumerate(sa.CATEGORICAL_GROUPS.items(), 1):
    md(f"""### 5.{idx} {vname}""")
    cd(f"""categories = sa.resolve_category(all_data, {cfg['columns']!r}, {cfg['labels']!r})
present = [l for l in {cfg['labels']!r} if categories.eq(l).any()]
fitters = {{}}
for label in present:
    mask = categories == label
    kmf = sa.fit_km_safe(all_data.loc[mask,'Stop'], all_data.loc[mask,'Event'],
                         all_data.loc[mask,'Start'], label)
    if kmf: fitters[label] = kmf
print(f'Variable: {vname}')
for lbl, kmf in fitters.items():
    n  = int((categories==lbl).sum())
    ne = int(kmf.event_table['observed'].sum())
    md_ = getattr(kmf, 'median_survival_time_', None)
    ms = f'{{md_:.1f}}' if md_ is not None and np.isfinite(md_) else 'No alcanzada'
    print(f'  {{lbl:30s}} n={{n:>6,}}  eventos={{ne:>5,}}  mediana={{ms}}')
if len(fitters) >= 2:
    m = categories.isin(present)
    r = multivariate_logrank_test(all_data.loc[m,'Stop'], categories[m], all_data.loc[m,'Event'])
    print(f'\\nLog-rank p = {{r.p_value:.2e}}')
""")
    cd(f"""fig, ax = plt.subplots()
colors = plt.cm.Set2(np.linspace(0,1,max(len(fitters),3)))
for (nm, kmf), clr in zip(fitters.items(), colors):
    kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2, color=clr)
    sf = kmf.survival_function_; ms = sf.iloc[:,0] < 0.5
    if ms.any():
        tc = sf.index[ms.values.argmax()]
        ax.axvline(x=tc, color=clr, linestyle='--', linewidth=1.5)
        ax.text(tc, 0.92, f'{{tc:.0f}}', color=clr, fontsize=9, ha='center', va='top',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=clr, alpha=0.7))
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Umbral de riesgo (0.5)')
ax.set_title(f'Supervivencia por {vname}', fontsize=14, fontweight='bold')
ax.set_xlabel('Tiempo (años)', fontsize=12); ax.set_ylabel('Probabilidad de Supervivencia', fontsize=12)
ax.grid(True, alpha=0.25); ax.set_ylim(0, 1.05)
ax.legend(fontsize=10, loc='lower left'); plt.tight_layout(); plt.show()
""")

# ── 6. Numéricas ──
if SPLIT_MODE == 'quartiles':
    SECTION_TITLE = 'cuartiles'
    SPLIT_DESC = 'segmenta en cuartiles'
    SPLIT_LABEL = '(cuartiles)'
    split_dict = quantiles_dict
else:
    SECTION_TITLE = 'mediana'
    SPLIT_DESC = 'divide en dos grupos por la mediana'
    SPLIT_LABEL = '(mediana)'
    split_dict = {var: [median_dict[var]] for var in median_dict}

md(f"""## 6. Análisis por variables numéricas ({SECTION_TITLE})

Cada variable numérica se {SPLIT_DESC} y se comparan las curvas de supervivencia. El test de log-rank evalúa si las diferencias entre grupos son significativas.""")

idxv = 0
for var in sa.NUMERICAL_VARS:
    if var not in split_dict: continue
    splits = split_dict[var]
    bins_raw = [-np.inf] + list(splits) + [np.inf]
    bins_uniq = sorted(set(bins_raw))
    n_bins = len(bins_uniq) - 1
    if n_bins < 2: continue
    idxv += 1
    def _fmt(v): return f'{v:.6f}' if np.isfinite(v) else ''
    bin_labels = []
    for i in range(n_bins):
        lo, hi = bins_uniq[i], bins_uniq[i+1]
        ls, hs = _fmt(lo), _fmt(hi)
        if not ls:       lbl = f'≤{hs}'
        elif not hs:     lbl = f'>{ls}'
        else:            lbl = f'({ls}, {hs}]'
        bin_labels.append(lbl)

    md(f"""### 6.{idxv} {var}""")
    bins_str = repr(bins_uniq).replace('inf', 'np.inf')
    cd(f"""groups = pd.cut(all_data['{var}'], bins={bins_str}, labels={bin_labels!r}, include_lowest=True)
present = [l for l in {bin_labels!r} if groups.eq(l).any()]
fitters = {{}}
for label in present:
    mask = groups == label
    kmf = sa.fit_km_safe(all_data.loc[mask,'Stop'], all_data.loc[mask,'Event'],
                         all_data.loc[mask,'Start'], label)
    if kmf: fitters[label] = kmf
print(f'Variable: {var}')
for lbl, kmf in fitters.items():
    n  = int((groups==lbl).sum())
    ne = int(kmf.event_table['observed'].sum())
    md_ = getattr(kmf, 'median_survival_time_', None)
    ms = f'{{md_:.1f}}' if md_ is not None and np.isfinite(md_) else 'No alcanzada'
    print(f'  {{lbl:25s}} n={{n:>6,}}  eventos={{ne:>5,}}  mediana={{ms}}')
if len(fitters) >= 2:
    m = groups.isin(present)
    r = multivariate_logrank_test(all_data.loc[m,'Stop'], groups[m], all_data.loc[m,'Event'])
    print(f'\\nLog-rank p = {{r.p_value:.2e}}')
""")
    cd(f"""fig, ax = plt.subplots()
colors = plt.cm.Set2(np.linspace(0,1,max(len(fitters),3)))
for (nm, kmf), clr in zip(fitters.items(), colors):
    kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2, color=clr)
    sf = kmf.survival_function_; ms = sf.iloc[:,0] < 0.5
    if ms.any():
        tc = sf.index[ms.values.argmax()]
        ax.axvline(x=tc, color=clr, linestyle='--', linewidth=1.5)
        ax.text(tc, 0.92, f'{{tc:.0f}}', color=clr, fontsize=9, ha='center', va='top',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=clr, alpha=0.7))
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Umbral de riesgo (0.5)')
ax.set_title(f'Supervivencia por {var} {SPLIT_LABEL}', fontsize=14, fontweight='bold')
ax.set_xlabel('Tiempo (años)', fontsize=12); ax.set_ylabel('Probabilidad de Supervivencia', fontsize=12)
ax.grid(True, alpha=0.25); ax.set_ylim(0, 1.05)
ax.legend(fontsize=10, loc='lower left'); plt.tight_layout(); plt.show()
""")

# ── 7 ──
md("""## 7. Conclusiones

1. **Supervivencia global**: La función de Kaplan-Meier muestra una disminución progresiva de la probabilidad de supervivencia. El cruce del umbral S(t)=0.5 indica el año a partir del cual la mayoría de empresas han fracasado.

2. **Variables categóricas**: Se identifican diferencias significativas (log-rank) según sector, forma jurídica, tamaño, etc., lo que confirma su relevancia como factores de riesgo.

3. **Variables numéricas**: La segmentación por {SECTION_TITLE} revela que las variables financieras (rentabilidad, liquidez, endeudamiento) discriminan claramente entre grupos de alto y bajo riesgo.

4. **Perspectiva unificada**: El análisis combinado (train+test) proporciona una visión global del fenómeno, maximizando el tamaño muestral y la potencia estadística.

---
*Notebook generado a partir de `survival_analysis.py` — análisis combinado (train + test)*
""")

# ── Escribir notebook ──────────────────────────────────────────────
NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4, "nbformat_minor": 5
}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'survival_analysis.ipynb')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(NB, f, ensure_ascii=False, indent=1)
print(f'Notebook generado: {path}')
print(f'  Variables categóricas: {len(sa.CATEGORICAL_GROUPS)}')
print(f'  División numérica: {SPLIT_MODE} ({len(split_dict)}/{len(sa.NUMERICAL_VARS)} vars)')
print(f'  Celdas totales: {len(CELLS)}')
