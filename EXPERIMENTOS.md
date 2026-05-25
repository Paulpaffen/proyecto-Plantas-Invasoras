# Bitácora de experimentos — detección de plantas invasoras

Registro de experimentos para comparar configuraciones de forma ordenada: log de
resultados y decisiones de cada corrida.

> **Estado al 2026-05-22:** Exp 0, Exp 1 y Exp 2 cerrados. Exp 2 (neg 0.20) confirma la
> conclusión del Exp 1: **RF-DETR es el mejor modelo en todas las métricas**; bajar negativos
> a 0.20 no recuperó Retamo (sigue dentro del ruido). Mejor checkpoint global: RF-DETR Exp 2.

---

## Configuración FIJA (igual en todos los experimentos salvo que se diga)

- **Clases (orden):** Acacia negra, Buchon de agua, Helecho de agua, Retamo espinoso.
- **Negativos:** `MODO_NEGATIVOS = "fondo"` (la clase `sin_invasion` se trata como fondo sin etiqueta).
- **Split:** compartido entre COCO y YOLO, estratificado, `seed=42`, `val_ratio=0.2`. Registrado en `split_manifest.json`.
- **Dataset origen:** `NuevasPlantas2.*` (337 imágenes: 195 positivas + 142 negativas).
- **Hardware:** AMD RX 9060 XT (16 GB), ROCm en Windows, kernel `Python 3.12 (ROCm — Radeon)`.
- **Metas del informe:** mAP50 > 0.70, mAP50-95 > 0.50, precisión > 0.75, recall > 0.70.

### Hiperparámetros por modelo

**RF-DETR** (`RFDETRBase`): `resolution=448`, `batch_size=2`, `grad_accumulation_steps=8`
(batch efectivo 16), `lr=1e-4`, `use_ema=False`, `amp=True`, `epochs=100`.
→ Config tuneada para no crashear el driver AMD. Se mantiene constante; la variable de
estudio es `NEG_RATIO_MAX`.

**RT-DETR** (`rtdetr-l.pt`, ultralytics): `freeze=1`, `lr0=1e-4`, `lrf=0.01`,
`warmup_epochs=3`, `epochs=100`, `amp=True`. Los parámetros que cambiaron entre Exp 0 y
Exp 1 están en cada experimento.

---

## Tabla comparativa (resumen)

| Exp | NEG_RATIO_MAX | Modelo  | mAP50 | mAP50-95 | Precisión | Recall | Notas |
|-----|---------------|---------|-------|----------|-----------|--------|-------|
| 0   | 0.30          | RT-DETR | 0.117 | 0.055    | 0.567     | 0.062  | **mal entrenado** (batch 2 / 480) — no representativo |
| 0   | 0.30          | RF-DETR | 0.623 | 0.289    | 0.641     | 0.567  | baseline funcional |
| 1   | 0.30          | RT-DETR | 0.274 | 0.146    | 0.333     | 0.303  | corregido (batch 8/640/patience 30) → **estable**; subió mucho vs Exp 0 pero < RF-DETR |
| 1   | 0.30          | RF-DETR | 0.549 | 0.271    | 0.754     | 0.491  | = baseline dentro del ruido (media run ~0.54, igual que Exp 0) |
| 2   | 0.20          | RT-DETR | 0.374 | 0.159    | 0.458     | 0.349  | mejor RT-DETR hasta ahora (subió vs Exp 1) pero sigue muy < RF-DETR |
| 2   | 0.20          | RF-DETR | 0.637 | 0.272    | 0.829     | 0.536  | **mejor corrida global**; precisión 0.83. Retamo 0.140 → no mejoró (ruido) |

---

## Exp 0 — Baseline inicial (CERRADO)

**Config:** `NEG_RATIO_MAX=0.30`. RT-DETR con `batch=2`, `imgsz=480`, `patience=10`.
Split: 222 train / 56 valid (83 negativos conservados, 59 descartados por el tope).

### RF-DETR (funcional) — mejor época 67
| Métrica | Valor |
|---|---|
| mAP50 | 0.6232 |
| mAP50-95 | 0.2885 |
| Precisión | 0.6411 |
| Recall | 0.5674 |

**AP por clase (mAP50-95):**
| Clase | AP |
|---|---|
| Helecho de agua | 0.427 |
| Buchon de agua | 0.303 |
| Acacia negra | 0.277 |
| **Retamo espinoso** | **0.147** ← clase más débil |

Observaciones:
- Hizo **plateau** hacia la época ~20-30; el resto oscila (set de validación chico = 56 imgs → métricas con varianza).
- **Retamo espinoso** es el punto débil claro. Hipótesis: era la clase sobre la que el modelo viejo colapsaba (falsos positivos), y los negativos pudieron sobre-corregirla. → motiva el Exp 2 (menos negativos).
- **Respaldo guardado:** `runs/rfdetr/baseline_v1_fondo030/` (checkpoints `.pth` + `metrics.csv` + README). Para inferencia: `checkpoint_best_regular.pth`.

### RT-DETR (NO representativo)
| Métrica | Valor |
|---|---|
| mAP50 | 0.1169 |
| mAP50-95 | 0.0547 |
| Precisión | 0.5671 |
| Recall | 0.0621 |

- Paró en la época 44 (early-stop, `patience=10`), nunca convergió.
- Métricas oscilando salvajemente entre épocas (precisión 0.58↔0.02), `cls_loss` estancado en ~1.4.
- **Causa:** `batch=2` (ruido brutal en el matching de DETR) + `imgsz=480`. No es límite del modelo, es config. → motiva el Exp 1.

---

## Exp 1 — RT-DETR corregido @ neg 0.30 (EN CURSO, llenar al despertar)

**Único cambio vs Exp 0:** hiperparámetros de RT-DETR. Datos idénticos (mismo split, neg 0.30).
- RT-DETR: `batch=8` (antes 2), `imgsz=640` (antes 480), `patience=30` (antes 10). Resto igual.
- RF-DETR: sin cambios (se reentrena solo porque se hizo Run All; debería reproducir ~0.62).

**Objetivo:** obtener un RT-DETR entrenado de verdad para una comparación **justa** RT-DETR vs RF-DETR (misma data, mismo valid).

**Resultados (tabla celda 5):**
| Modelo | mAP50 | mAP50-95 | Precisión | Recall |
|---|---|---|---|---|
| RT-DETR | 0.2738 | 0.1457 | 0.3328 | 0.3030 |
| RF-DETR | 0.5489 | 0.2712 | 0.7536 | 0.4907 |

**RT-DETR — el fix funcionó (éxito):**
- 93 épocas (early-stop patience 30), pico mAP50 ~0.35 (época 80).
- **Ahora estable**: std de mAP50 en las últimas 10 épocas = 0.015 (en Exp 0 saltaba 0.58↔0.02).
- mAP50 0.117→0.274, recall 0.062→0.303. Entrena de verdad; sigue < RF-DETR por dataset pequeño + backbone más débil que DINOv2. **Resultado legítimo y justo.**

**RF-DETR — NO empeoró, es RUIDO de validación (hallazgo clave):**
- Run nuevo: mejor época 11 → mAP50 0.549. Baseline: mejor época 67 → mAP50 0.623.
- Pero **la media del run es idéntica**: nuevo 0.543 (std 0.118) vs baseline 0.547 (std 0.095). Mismo modelo, mismo rendimiento.
- La diferencia 0.549 vs 0.623 es azar: con **56 imágenes de validación**, el mAP50 oscila ±0.10 entre épocas, y elegir "la mejor época por mAP50-95" da picos distintos cada corrida (el run nuevo incluso tocó 0.694 en otras épocas).
- AP por clase (nuevo): Acacia 0.36, Helecho 0.32, Buchon 0.30, **Retamo 0.114** (sigue siendo la peor, como en el baseline 0.147).

**CONCLUSIÓN del Exp 1:**
1. El problema no es la config sino el **tamaño del dataset**: 56 imgs de valid → métricas con ruido enorme (±0.10). No fiarse de un solo número de una sola corrida.
2. **Para entrega/inferencia usar el checkpoint del baseline** (`baseline_v1_fondo030`, mAP50 0.623) — es el guardado más afortunado.
3. **Más datos** es la palanca #1, por encima de cualquier hiperparámetro.
4. La diferencia de Retamo (0.114 vs 0.147) también cae dentro del ruido → en el Exp 2 buscar mejora **consistente**, no un pico.

---

## Exp 2 — Menos negativos @ neg 0.20 (CERRADO)

**Único cambio vs Exp 1:** `NEG_RATIO_MAX = 0.20` en la celda 1 (reejecutar desde celda 2).
Esto rebaja los negativos (de ~30% a ~20% del total) → menos supresión.

**Hipótesis:** sube el recall, especialmente en **Retamo espinoso** (comparar su AP contra el 0.147 del baseline). Riesgo: pueden subir los falsos positivos (baja precisión).

**Cuidado de comparación:** al cambiar el tope cambia el set de validación, así que NO es
directamente comparable contra Exp 0/1 a nivel de "mismas imágenes". Lo válido:
- RT-DETR vs RF-DETR **dentro** del Exp 2 (mismo valid) → comparación justa.
- Tendencia del AP de Retamo y del recall global vs experimentos previos → señal, con algo de ruido por el valid distinto.

**Resultados (tabla celda 5):**
| Modelo | mAP50 | mAP50-95 | Precisión | Recall |
|---|---|---|---|---|
| RT-DETR | 0.3741 | 0.1588 | 0.4579 | 0.3494 |
| RF-DETR | 0.6368 | 0.2719 | 0.8286 | 0.5362 |

**RF-DETR — mejor época 34 (lectura de `metrics.csv`):**
- Es la **mejor corrida de todo el proyecto**: mAP50 0.637 y, sobre todo, **precisión 0.829** (la más alta vista, por encima del baseline 0.641 y del Exp 1 0.754). Cumple ya la meta de precisión (>0.75).
- AP por clase (mAP50-95): Acacia 0.328, Buchon 0.321, Helecho 0.299, **Retamo 0.140**.
- **La hipótesis de Retamo NO se cumplió:** 0.140 vs 0.147 (Exp 0) / 0.114 (Exp 1) → sigue siendo la peor clase y la diferencia cae dentro del ruido. Bajar negativos a 0.20 **no la recuperó**.
- Recall 0.536: prácticamente igual al baseline (0.567), no subió como esperaba la hipótesis. Confirma que el cuello de botella es el **tamaño del dataset**, no el ratio de negativos.

**RT-DETR — su mejor corrida, pero sigue muy por debajo:**
- mAP50 0.374 (vs 0.274 en Exp 1, 0.117 en Exp 0): la tendencia al alza continúa, el modelo entrena de verdad.
- Aun así pierde contra RF-DETR en **todas** las métricas, y por mucho (precisión 0.46 vs 0.83, recall 0.35 vs 0.54).

**CONCLUSIÓN del Exp 2:**
1. **RF-DETR gana limpio** en mAP50, mAP50-95, precisión y recall. Es el modelo recomendado para la entrega.
2. Bajar negativos 0.30→0.20 **subió la precisión** de RF-DETR (0.75→0.83) **sin hundir el recall** (0.49→0.54). Por la regla práctica de negativos (ver abajo), **0.20 gana** sobre 0.30 para este modelo.
3. Retamo espinoso sigue siendo el punto débil y **no** se arregla con menos negativos → necesita **más datos** de esa clase (palanca #1, ya señalada en Exp 1).
4. Checkpoint a usar para inferencia/entrega: el de **RF-DETR Exp 2**, respaldado en `runs/rfdetr/exp2_neg020/checkpoint_best_regular.pth` (copia segura; `runs/rfdetr/plantas_invasoras/` se sobrescribe en el próximo Run All).

---

## Qué comparar entre 0.30 y 0.20

1. **Recall global** y **AP de Retamo espinoso** (¿el menos-negativos lo recupera?).
2. **Precisión** (¿reaparecen falsos positivos al quitar negativos?).
3. **mAP50 / mAP50-95** generales de cada modelo.
4. Estabilidad del entrenamiento (RT-DETR sobre todo).
5. Regla práctica de negativos: si el recall sube mucho pero la precisión se hunde, 0.30 era mejor; si el recall sube sin perder precisión, 0.20 gana.

---

## Artefactos y ubicaciones

| Qué | Dónde |
|---|---|
| Baseline RF-DETR funcional (Exp 0) | `runs/rfdetr/baseline_v1_fondo030/` |
| **RF-DETR Exp 2 (mejor global, recomendado entrega)** | `runs/rfdetr/exp2_neg020/` (checkpoint_best_regular.pth + metrics.csv + README) |
| Split reproducible | `split_manifest.json` |
| RT-DETR del run activo | `runs/rtdetr/plantas_invasoras/` (results.csv, weights/best.pt) |
| RF-DETR del run activo | `runs/rfdetr/plantas_invasoras/` (metrics.csv, checkpoint_best_regular.pth) |
| Datasets derivados (se reconstruyen) | `data_rtdetr/`, `data_rfdetr/` |
| Lectura de métricas | RT-DETR → `results.csv`; RF-DETR → `metrics.csv` (mejor época por `val/mAP_50_95`) |
