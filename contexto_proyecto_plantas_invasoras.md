# Proyecto: Detección de plantas invasoras en Cundinamarca, Colombia

## Contexto general

Proyecto universitario cuyo objetivo es construir un modelo de visión computacional capaz de detectar plantas invasoras presentes en el departamento de Cundinamarca, Colombia.

El equipo está dividiendo el trabajo:
- Un compañero está probando **YOLOv8/v11** con el mismo dataset
- Este entorno (Claude Code) se encarga de explorar y entrenar con **RT-DETR** y **RF-DETR**

El objetivo final es comparar los tres modelos (YOLO, RT-DETR, RF-DETR) con las mismas métricas y el mismo dataset para determinar cuál funciona mejor para esta tarea.

---

## Modelo de referencia

Se tomó como inspiración el siguiente proyecto de Roboflow Universe:
- **URL:** https://universe.roboflow.com/invasive-plants/invasive-plants-lak90
- Este proyecto tiene imágenes y anotaciones de plantas invasoras (en otro contexto geográfico)
- Se puede descargar su **dataset** para combinarlo con el dataset propio de Cundinamarca

---

## Estrategia: Transfer Learning en dos niveles

```
COCO (millones de imágenes)
    → pesos preentrenados RT-DETR / RF-DETR
        → dataset Roboflow Universe + dataset propio de Cundinamarca
            → modelo final de detección
```

### ¿Por qué transfer learning?
- Los modelos RT-DETR y RF-DETR ya fueron preentrenados con COCO (80 categorías, millones de imágenes)
- El modelo ya "sabe ver": bordes, texturas, formas, colores
- Solo se necesita afinar el *head* del modelo para las clases nuevas (especies de plantas invasoras)
- Esto permite entrenar con datasets pequeños y en menos tiempo

### Etapas del proceso
1. **Modelo base** — descargar RT-DETR o RF-DETR con pesos preentrenados en COCO
2. **Congelar capas** — el backbone se congela, solo se entrena el head de detección
3. **Dataset propio** — imágenes de plantas invasoras de Cundinamarca, anotadas con bounding boxes
4. **Fine-tuning** — entrenar con el dataset combinado, pocas épocas
5. **Evaluación** — medir mAP, precisión y recall por especie

---

## Modelos a usar

### RT-DETR
- Arquitectura basada en transformers, tiempo real
- Elimina NMS (Non-Maximum Suppression), lo que mejora velocidad
- Alta precisión, especialmente para minimizar falsos positivos
- Implementado en la librería `ultralytics`
- Licencia: Apache 2.0 — **completamente gratuito**

**Instalación:**
```bash
pip install ultralytics
```

**Variantes disponibles** (elegir según VRAM de la GPU):
| Variante | Parámetros | VRAM aprox. |
|---|---|---|
| rtdetr-l.pt | 32M | ~6 GB |
| rtdetr-x.pt | 67M | ~10 GB |

### RF-DETR
- Desarrollado por Roboflow, backbone DINOv2
- Elimina anchors tradicionales y NMS
- 54.7% mAP con menos de 5ms de latencia
- Integración nativa con el ecosistema Roboflow
- Licencia: open source — **completamente gratuito**

**Instalación:**
```bash
pip install rfdetr
```

---

## Dataset

### Dataset propio (Cundinamarca)
- Imágenes recolectadas por el equipo en campo
- Plantas invasoras del departamento de Cundinamarca, Colombia
- Anotadas con bounding boxes por especie

### Dataset de Roboflow Universe (complementario)
- Proyecto: https://universe.roboflow.com/invasive-plants/invasive-plants-lak90
- Se descarga con la API de Roboflow (requiere API key gratuita)

**Descarga del dataset de Roboflow Universe:**
```python
from roboflow import Roboflow

rf = Roboflow(api_key="TU_API_KEY")
proyecto = rf.workspace("invasive-plants").project("invasive-plants-lak90")
dataset = proyecto.version(1).download("yolov8")  # formato compatible con ultralytics
```

### Herramientas de anotación (gratuitas)
- **Roboflow** (tier gratuito) — anotar, aumentar datos y exportar
- **Label Studio** — open source, 100% local

### Data augmentation recomendada
Roboflow o `albumentations` en Python:
- Rotación aleatoria
- Cambios de brillo/contraste
- Flip horizontal
- Recorte aleatorio
- Simulación de distintas condiciones de luz (importante para campo)

---

## Entrenamiento local con GPU

Todo el entrenamiento se realiza **localmente** en el PC del equipo con GPU. No se usa la nube de pago de Roboflow ni ningún servicio de entrenamiento externo. Stack 100% gratuito.

### Requisitos
- Python 3.10+
- PyTorch con soporte CUDA
- GPU Nvidia con CUDA instalado

### Verificar que PyTorch detecta la GPU
```python
import torch
print(torch.cuda.is_available())      # debe imprimir True
print(torch.cuda.get_device_name(0))  # nombre de la GPU
```

---

## Código base: RT-DETR con ultralytics

### Entrenamiento
```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")  # descarga pesos preentrenados automáticamente

results = model.train(
    data="dataset/data.yaml",   # ruta al archivo de configuración del dataset
    epochs=50,
    imgsz=640,
    batch=8,                    # reducir si hay problemas de VRAM
    device=0,                   # GPU 0
    freeze=1,                   # congela el backbone (transfer learning)
    lr0=0.0001,                 # learning rate bajo para fine-tuning
    project="runs/rtdetr",
    name="plantas_invasoras"
)
```

### Evaluación
```python
metrics = model.val()
print(metrics.box.map)    # mAP50-95
print(metrics.box.map50)  # mAP50
```

### Inferencia (probar con una imagen nueva)
```python
results = model.predict("imagen_prueba.jpg", conf=0.25)
results[0].show()
```

---

## Código base: RF-DETR

### Entrenamiento
```python
from rfdetr import RFDETRBase
from rfdetr.util.coco_utils import get_coco_api_from_dataset

model = RFDETRBase(pretrain_weights="rf-detr-base.pth")

model.train(
    dataset_dir="dataset/",     # carpeta con train/, valid/, test/ y _annotations.coco.json
    epochs=50,
    batch_size=8,
    grad_accumulation_steps=1,
    lr=1e-4,
    output_dir="runs/rfdetr/plantas_invasoras"
)
```

> **Nota:** RF-DETR espera el dataset en formato COCO (JSON), no YOLO. Al exportar desde Roboflow elegir formato "COCO".

### Evaluación
```python
metrics = model.val(dataset_dir="dataset/")
```

---

## Estructura de carpetas recomendada

```
proyecto_plantas/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── runs/
│   ├── rtdetr/
│   └── rfdetr/
├── train_rtdetr.py
├── train_rfdetr.py
└── evaluar.py
```

### Ejemplo de data.yaml (para RT-DETR / ultralytics)
```yaml
path: ./dataset
train: train/images
val: valid/images
test: test/images

nc: 3  # número de clases (ajustar según las especies)
names:
  - especie_1
  - especie_2
  - especie_3
```

---

## Métricas de evaluación

| Métrica | Descripción | Meta sugerida |
|---|---|---|
| mAP50 | Precisión media al umbral IoU 0.5 | > 0.70 |
| mAP50-95 | Precisión media en rangos de IoU | > 0.50 |
| Precisión | De las detecciones, ¿cuántas son correctas? | > 0.75 |
| Recall | De las plantas reales, ¿cuántas detectó? | > 0.70 |

Para el informe universitario se deben comparar estas métricas entre RT-DETR, RF-DETR y el YOLO del compañero, usando exactamente el mismo dataset de validación.

---

## Notas importantes

- **Todo es gratuito:** ultralytics, rfdetr, roboflow (tier gratuito para anotar), PyTorch — ningún costo
- **No usar la nube de Roboflow para entrenar** — solo para anotar y exportar el dataset
- El compañero usa YOLO con el mismo dataset — usar `data.yaml` idéntico para comparación justa
- Guardar siempre los pesos del mejor modelo (`best.pt`) que ultralytics genera automáticamente en `runs/`
- Si hay problemas de VRAM, reducir `batch` o usar `rtdetr-l.pt` en vez de `rtdetr-x.pt`
