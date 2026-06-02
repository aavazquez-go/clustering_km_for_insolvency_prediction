# Estratificación de Riesgo de Insolvencia en PYMEs mediante Clustering K-Means y Análisis de Supervivencia Kaplan-Meier

Integración de **aprendizaje no supervisado (K-Means)** con **análisis de supervivencia no paramétrico (Kaplan-Meier)** para identificar perfiles de riesgo financiero (arquetipos) en PYMEs españolas.

## Pipeline

```
Dataset colapsado (117.052 PYMEs)
        │
        ▼
Selección de 26 ratios financieros (F16–F43)
        │
        ▼
Estandarización (z-score) + filtro de outliers (percentil 99)
        │
        ▼
Determinación de K óptimo (codo + silhouette, K=2..10)
        │
        ▼
K-Means → arquetipos financieros
        │
        ▼
Perfilado (medias por ratio y arquetipo)
        │
        ▼
Kaplan-Meier estratificado + test log-rank
        │
        ▼
Gráficos: elbow, perfiles, curvas de supervivencia
```

## Resultados clave

| Métrica | Valor |
|---------|-------|
| Muestra | 117.052 PYMEs |
| K óptimo (silhouette) | **2** |
| Log-rank p-valor | **3.78 × 10⁻³⁰³** |

Dos arquetipos: **Resiliente** (67.7%, mediana no alcanzada > 20 años) y **Vulnerable** (32.3%, mediana 21 años).

## Estructura

```
├── archetype_clustering.py   # Clustering + supervivencia integrados
├── survival_analysis.py      # Análisis KM completo (base, categóricas, numéricas)
├── build_notebook.py         # Genera survival_analysis.ipynb
├── kaplan-meier.ipynb        # Notebook ejecutable (base)
├── survival_analysis.ipynb   # Notebook generado (combinado train+test)
├── datasets/
│   ├── train_set.csv         # Counting-process original (train)
│   ├── test_set.csv          # Counting-process original (test)
│   ├── train_collapsed.csv   # Colapsado 1 fila/empresa (train)
│   ├── test_collapsed.csv    # Colapsado 1 fila/empresa (test)
│   └── all_collapsed.csv     # Combinado train+test
├── plots/                    # Gráficos generados (45+ archivos PNG)
├── logs/                     # Logs detallados
└── docs/
    ├── solucion_clustering_arquetipos.md   # Documentación técnica del método
    └── resultados_arquetipos.md            # Resultados y conclusiones
```

## Requisitos

```
numpy pandas matplotlib scikit-learn lifelines
```

## Uso

```bash
# Análisis de arquetipos (clustering + supervivencia)
python3 archetype_clustering.py

# Especificar K manualmente
python3 archetype_clustering.py --k 3

# Análisis de supervivencia completo (train y test)
python3 survival_analysis.py

# Análisis combinado (train+test) con división por cuartiles
python3 survival_analysis.py --mode combined --split quartiles

# Generar notebook
python3 build_notebook.py
```

## Publicación

Si utiliza este código en una investigación académica, por favor cite el trabajo correspondiente (tesis doctoral / paper).
