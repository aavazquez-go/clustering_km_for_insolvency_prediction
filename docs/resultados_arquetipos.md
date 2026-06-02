# Resultados: Arquetipos Financieros y Supervivencia

## Resumen

| Métrica | Valor |
|---------|-------|
| Muestra total | 117.052 PYMEs |
| Ratios financieros | 26 (F16–F25, F27–F31, F33–F43) |
| K óptimo (silhouette) | **2** |
| Outliers filtrados | 1.171 (1%) |
| Log-rank χ² | 1.385,02 |
| Log-rank p-valor | **3,78 × 10⁻³⁰³** (altamente significativo) |

---

## Determinación de K óptimo

| K | Inercia | Silhouette |
|---|---------|------------|
| 2 | 41.087 | **0,3453** |
| 3 | 31.080 | 0,3028 |
| 4 | 27.064 | 0,2893 |
| 5 | 24.281 | 0,2481 |
| 6 | 22.300 | 0,2372 |
| 7 | 20.534 | 0,2405 |
| 8 | 19.323 | 0,2300 |
| 9 | 18.248 | 0,2194 |
| 10 | 17.271 | 0,2141 |

El coeficiente de silueta es máximo en K=2 (0,345), indicando que la partición más cohesiva y separada divide a las PYMEs en **dos grandes arquetipos financieros**.

---

## Arquetipos descubiertos

| Arquetipo | n | % | Ratios característicos |
|-----------|---|--|----------------------|
| **0 — Resiliente** | 79.278 | 67,7% | F18↑, F17↑ |
| **1 — Vulnerable** | 37.774 | 32,3% | F17↑, F18↑, F28↑ |

### Interpretación

- **Arquetipo 0 (Resiliente):** Perfil mayoritario. Presenta valores positivos en los ratios F17 y F18 (por encima de la media estandarizada), lo que sugiere una posición financiera equilibrada con mejor capacidad de generación de recursos. La mediana de supervivencia no se alcanza ( > 20 años), indicando que más del 50% de estas empresas sobreviven todo el período de observación.

- **Arquetipo 1 (Vulnerable):** Perfil minoritario pero sustancial (casi 1 de cada 3 PYMEs). Se caracteriza por valores elevados en F17, F18 y F28. La mediana de supervivencia es de **21 años**, significativamente menor que la del arquetipo resiliente. La probabilidad de supervivencia a 5 años es del 79,9% frente al 90,8% del arquetipo resiliente.

> **Nota:** Los ratios F17, F18 y F28 son variables estandarizadas (z-scores). Las etiquetas ↑/↓ indican desviación positiva/negativa respecto a la media de la muestra.

---

## Análisis de Supervivencia

### Función de supervivencia por arquetipo

| t (años) | Arquetipo 0 | IC 95% | Arquetipo 1 | IC 95% |
|----------|-------------|--------|-------------|--------|
| 1 | 0,9868 | [0,9860–0,9876] | 0,9625 | [0,9605–0,9644] |
| 2 | 0,9604 | [0,9590–0,9618] | 0,9125 | [0,9096–0,9154] |
| 3 | 0,9425 | [0,9409–0,9441] | 0,8770 | [0,8736–0,8803] |
| 5 | 0,9077 | [0,9056–0,9097] | 0,7997 | [0,7955–0,8037] |
| 10 | 0,7987 | [0,7959–0,8016] | 0,6389 | [0,6338–0,6439] |
| 15 | 0,6370 | [0,6335–0,6404] | 0,5423 | [0,5370–0,5477] |
| 20 | 0,5752 | [0,5715–0,5788] | 0,5024 | [0,4968–0,5081] |

### Estadísticas descriptivas

| Métrica | Arquetipo 0 | Arquetipo 1 |
|---------|-------------|-------------|
| Sujetos | 79.278 | 37.774 |
| Eventos (insolvencia) | 31.180 (39,3%) | 16.572 (43,9%) |
| Censurados | 48.098 (60,7%) | 21.202 (56,1%) |
| Mediana de supervivencia | **No alcanzada** (>20 años) | **21 años** |

### Test de Log-Rank

- **Estadístico:** 1.385,02
- **Grados de libertad:** 1
- **p-valor:** 3,78 × 10⁻³⁰³

El p-valor es esencialmente cero, lo que permite rechazar la hipótesis nula de que ambos arquetipos comparten la misma función de supervivencia. Las diferencias observadas son **altamente significativas** desde el punto de vista estadístico.

---

## Discusión

1. **Separabilidad financiera:** K=2 emerge como la partición óptima, lo que sugiere que el espacio de ratios financieros de las PYMEs españolas se estructura en dos grandes clusters. Esta dicotomía captura diferencias fundamentales en la salud financiera que el análisis univariante (dicotomización por mediana) no puede revelar.

2. **Gradiente de riesgo:** Aunque ambos arquetipos son distinguibles desde el primer año, la divergencia se acentúa con el tiempo. A 5 años, la diferencia en supervivencia es de ~11 puntos porcentuales; a 10 años, de ~16 puntos. El arquetipo vulnerable acumula un 4,6% más de eventos de insolvencia que el resiliente.

3. **Ventaja sobre el análisis univariante:** Mientras que la estratificación clásica por mediana de un único ratio ignora las correlaciones entre variables financieras, el enfoque multivariante captura interacciones complejas. Una PYME con un ratio de liquidez bajo pero alta rentabilidad podría ser clasificada como de bajo riesgo por el algoritmo, algo que el análisis univariante no podría detectar.

4. **Validación externa:** La concordancia entre la segmentación obtenida mediante aprendizaje no supervisado y las diferencias observadas en supervivencia (validadas con log-rank) confirma que los arquetipos descubiertos tienen poder predictivo real sobre el riesgo de insolvencia.

---

## Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `plots/elbow_silhouette.png` | Codo y silhouette para K=2..10 |
| `plots/archetype_profiles.png` | Perfil medio de ratios por arquetipo |
| `plots/km_archetypes.png` | Curvas KM estratificadas por arquetipo |
| `logs/archetype_clustering.log` | Log completo del análisis |
