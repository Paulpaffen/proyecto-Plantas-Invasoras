# SETUP — Detección de plantas invasoras

Guía rápida para clonar el repo y dejarlo listo para entrenar. Los datasets y los pesos **no están en Git** (pesan demasiado); se descargan aparte y se ubican en las rutas que describe este documento.

---

## 1 · Clonar el repo

```powershell
git clone <URL_DEL_REPO> "im y vi"
cd "im y vi"
```

## 2 · Crear entorno e instalar dependencias

```powershell
pip install ultralytics
pip install "rfdetr[train,loggers]"
```

> El notebook (`entrenamiento_plantas.ipynb`) ya tiene una celda inicial con los `pip install`, así que también puedes saltarte este paso y ejecutar la celda 0.

## 3 · Descargar los datasets

Los datasets se exportan desde Roboflow en **dos formatos** (uno por modelo). Se descargan aparte y se colocan en la **raíz del proyecto** (al mismo nivel que `entrenamiento_plantas.ipynb`).

### 3.1 · Dataset COCO (para RF-DETR)

- Formato Roboflow: **COCO**
- Carpeta destino: `Plantas.coco/`
- Estructura esperada:

```
Plantas.coco/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/                  ← si no existe, el notebook lo crea (split 80/20 desde train)
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/                   ← opcional
    ├── _annotations.coco.json
    └── *.jpg
```

> El notebook arregla automáticamente el bug de Roboflow (bbox como strings) y crea el split de validación si falta.

### 3.2 · Dataset YOLOv8-OBB (para RT-DETR / comparación con YOLO)

- Formato Roboflow: **YOLOv8 Oriented Bounding Boxes**
- Carpeta destino: `Plantas.yolov8-obb/`
- Estructura esperada:

```
Plantas.yolov8-obb/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/                  ← idealmente con split propio
    ├── images/
    └── labels/
```

Contenido mínimo de `data.yaml`:

```yaml
path: Plantas.yolov8-obb        # relativo a la raíz del proyecto
train: train/images
val: valid/images               # si todavía no hay valid/, dejar train/images temporalmente

nc: 4
names:
  - Acacia negra
  - Buchon de agua
  - Helecho de agua
  - Retamo espinoso
```

> El notebook (celda 1) configura `ultralytics.settings.datasets_dir = ROOT` para que la línea `path: Plantas.yolov8-obb` se resuelva relativa a la carpeta del proyecto, sin importar desde dónde se ejecute el notebook.

### 3.3 · Carpeta de imágenes sueltas para inferencia (opcional)

Si quieres probar la celda 6 (Inferencia), pon imágenes en:

```
Entrenamiento IyV/
└── acacia_negra/
    └── acacia_negra_000001.jpg
```

O simplemente cambia la variable `IMAGEN` en la celda 6 por la ruta a cualquier foto.

## 4 · Descargar los pesos preentrenados

Se descargan automáticamente la primera vez que se ejecuta cada modelo, pero también puedes ponerlos a mano en la raíz del proyecto:

| Archivo            | Modelo          | Se descarga al ejecutar... |
| ------------------ | --------------- | -------------------------- |
| `rtdetr-x.pt`      | RT-DETR (Ultralytics) | celda 3 del notebook |
| `rf-detr-base.pth` | RF-DETR (Roboflow)    | celda 4 del notebook |

## 5 · Estructura final esperada del proyecto

```
im y vi/
├── entrenamiento_plantas.ipynb     ← flujo principal (sí va a Git)
├── contexto_proyecto_plantas_invasoras.md
├── SETUP.md                        ← este archivo
├── .gitignore
│
├── Plantas.coco/                   ← (no va a Git)
├── Plantas.yolov8-obb/             ← (no va a Git)
├── Entrenamiento IyV/              ← imágenes sueltas (no va a Git)
│
├── rtdetr-x.pt                     ← (no va a Git)
├── rf-detr-base.pth                ← (no va a Git)
│
└── runs/                           ← salidas de entrenamiento (no va a Git)
    ├── rtdetr/plantas_invasoras/
    │   └── weights/best.pt
    └── rfdetr/plantas_invasoras/
        ├── checkpoint_best_regular.pth
        └── metrics.csv
```

## 6 · Ejecutar el notebook

Abrir `entrenamiento_plantas.ipynb` y correr las celdas en orden:

0. Instalación de dependencias
1. Verificar GPU + setup de rutas (`ROOT`, `COCO_DIR`, `YOLO_YAML`, ...)
2. Preparar dataset RF-DETR (fix de bboxes + split de validación)
3. Entrenamiento RT-DETR
4. Entrenamiento RF-DETR
5. Evaluación y tabla comparativa
6. Inferencia sobre una imagen

Las secciones 3 y 4 son independientes — puedes correr solo una.

## 7 · Notas

- **Sin GPU NVIDIA** entrena en CPU; cada época puede tardar varios minutos. Reducir `epochs`, `batch` o cambiar `rtdetr-x.pt` por `rtdetr-l.pt` ayuda.
- La métrica de RF-DETR en la celda 5 se lee del `metrics.csv` que Lightning va guardando durante el entrenamiento (toma la mejor época por `mAP_50_95`). `RFDETRBase` no tiene método `.val()`.
- Si cambias de máquina, basta con copiar las carpetas `Plantas.coco/`, `Plantas.yolov8-obb/` y los `.pth/.pt` — el notebook no depende de ninguna ruta absoluta hardcodeada.
