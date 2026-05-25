# SETUP — Detección de plantas invasoras

Guía rápida para clonar el repo y dejarlo listo para entrenar. Los datasets y los pesos **no están en Git** (pesan demasiado); se descargan aparte y se ubican en las rutas que describe este documento.

---

## 1 · Clonar el repo

```powershell
git clone <URL_DEL_REPO> "im y vi"
cd "im y vi"
```

## 2 · Crear entorno e instalar dependencias

### 2.A · Opción CPU o GPU NVIDIA (sencillo)

```powershell
pip install ultralytics
pip install "rfdetr[train,loggers]"
```

> El notebook (`entrenamiento_plantas.ipynb`) ya tiene una celda inicial con los `pip install`, así que también puedes saltarte este paso y ejecutar la celda 0.

### 2.B · Opción GPU AMD Radeon (RX 7000 / RX 9000 / Ryzen AI 300 / Ryzen AI MAX) en Windows

Las wheels oficiales de PyTorch-ROCm para Windows solo existen para **Python 3.12** (no 3.13). El driver Adrenalin debe ser 25.10.2 o superior. Pasos:

```powershell
# 1. Crear venv con Python 3.12 dentro del proyecto
py -3.12 -m venv .venv-rocm
.\.venv-rocm\Scripts\python.exe -m pip install --upgrade pip

# 2. Instalar PyTorch ROCm (~2 GB de descarga)
.\.venv-rocm\Scripts\python.exe -m pip install --find-links https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/ `
  torch==2.9.1+rocm7.2.1 torchvision==0.24.1+rocm7.2.1 torchaudio==2.9.1+rocm7.2.1

# 3. Instalar el resto en el mismo venv
.\.venv-rocm\Scripts\python.exe -m pip install ultralytics "rfdetr[train,loggers]" ipykernel pandas

# 4. Registrar el venv como kernel de Jupyter
.\.venv-rocm\Scripts\python.exe -m ipykernel install --user --name rocm `
  --display-name "Python 3.12 (ROCm — Radeon)"

# 5. Verificar
.\.venv-rocm\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

En el notebook: **Kernel → Change kernel → "Python 3.12 (ROCm — Radeon)"**. La celda 1 detecta automáticamente la GPU y pone `DEVICE = 0`; el resto del notebook usa esa variable. Si no aparece el kernel en VS Code/Jupyter, reinicia la app.

> Aunque la API se llama `torch.cuda.*`, internamente se mapea a ROCm — esto es a propósito para que el código portado de NVIDIA funcione sin cambios.

## 3 · Descargar los datasets

Los datasets se exportan desde Roboflow en **dos formatos** (uno por modelo). Se descargan aparte y se colocan en la **raíz del proyecto** (al mismo nivel que `entrenamiento_plantas.ipynb`). Estos exports quedan **intactos**: la celda 2 / `preparar_datos.py` construye a partir de ellos los datasets derivados `data_rfdetr/` y `data_rtdetr/` (con split de validación compartido), sin tocar los originales.

> **Clases:** el dataset trae las 4 invasoras (`Acacia negra`, `Buchon de agua`, `Helecho de agua`, `Retamo espinoso`) y una 5ª clase `sin_invasion` (plantas NO invasoras) que se usa como **negativos**. Dos variables (celda 1 del notebook / arriba de los scripts) controlan esto:
> - `MODO_NEGATIVOS`: `"fondo"` (recomendado — los pasa a negativos sin etiqueta, `nc=4`) o `"clase"` (los mantiene como 5ª clase, `nc=5`).
> - `NEG_RATIO_MAX`: tope de proporción de negativos en el dataset final (por defecto `0.30`; `1.0` = conservarlos todos). Demasiados negativos bajan el recall.
>
> Como todo se reconstruye desde los exports pristinos, cambiar de modo es instantáneo y no destructivo (no hace falta re-exportar de Roboflow).

### 3.1 · Dataset COCO (para RF-DETR)

- Formato Roboflow: **COCO**
- Carpeta destino: `NuevasPlantas2.coco/`
- Estructura esperada:

```
NuevasPlantas2.coco/
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

### 3.2 · Dataset YOLOv8 (para RT-DETR / comparación con YOLO)

- Formato Roboflow: **YOLOv8** (detección, cajas normales — no OBB)
- Carpeta destino: `NuevasPlantas2.yolov8/`
- Estructura esperada:

```
NuevasPlantas2.yolov8/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/                  ← idealmente con split propio
    ├── images/
    └── labels/
```

No hace falta editar `data.yaml` a mano: la celda 2 del notebook (y los scripts) lo **reescriben automáticamente** con `escribir_data_yaml()`, ajustando `path:`, `val:` y las clases según `MODO_NEGATIVOS`. El resultado en modo `"fondo"` es:

```yaml
path: NuevasPlantas2.yolov8     # relativo a la raíz del proyecto
train: train/images
val: valid/images               # train/images si todavía no hay valid/

nc: 4
names:
  - Acacia negra
  - Buchon de agua
  - Helecho de agua
  - Retamo espinoso
# en modo "clase" se añade un 5º nombre: sin_invasion (nc: 5)
```

> El notebook (celda 1) configura `ultralytics.settings.datasets_dir = ROOT` para que la línea `path: NuevasPlantas2.yolov8` se resuelva relativa a la carpeta del proyecto, sin importar desde dónde se ejecute el notebook.

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
| `rtdetr-l.pt`      | RT-DETR (Ultralytics) | celda 3 del notebook |
| `rf-detr-base.pth` | RF-DETR (Roboflow)    | celda 4 del notebook |

> El notebook usa `rtdetr-l.pt` (32M params, ~6 GB VRAM) para caber en 16 GiB. `rtdetr-x.pt` (67M, ~10 GB) es la opción más pesada si tienes VRAM de sobra.

## 5 · Estructura final esperada del proyecto

```
im y vi/
├── entrenamiento_plantas.ipynb     ← flujo principal (sí va a Git)
├── README.md
├── SETUP.md                        ← este archivo
├── EXPERIMENTOS.md
├── .gitignore
│
├── NuevasPlantas2.coco/            ← export Roboflow, pristino (no va a Git)
├── NuevasPlantas2.yolov8/          ← export Roboflow, pristino (no va a Git)
├── data_rfdetr/                    ← derivado COCO, lo crea la celda 2 (no va a Git)
├── data_rtdetr/                    ← derivado YOLO, lo crea la celda 2 (no va a Git)
├── split_manifest.json            ← split reproducible (SÍ va a Git — compártelo con el compañero)
├── Entrenamiento IyV/              ← imágenes sueltas (no va a Git)
│
├── rtdetr-l.pt                     ← (no va a Git)
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
1. Verificar GPU + setup de rutas, `MODO_NEGATIVOS` y `NEG_RATIO_MAX`
2. Construir datasets derivados `data_rfdetr/` y `data_rtdetr/` (split compartido + manejo de negativos + manifiesto)
3. Entrenamiento RT-DETR
4. Entrenamiento RF-DETR
5. Evaluación y tabla comparativa
6. Inferencia sobre una imagen

Las secciones 3 y 4 son independientes — puedes correr solo una.

## 7 · Notas

- **Sin GPU NVIDIA** entrena en CPU; cada época puede tardar varios minutos. Reducir `epochs`, `batch` o `imgsz` ayuda.
- La métrica de RF-DETR en la celda 5 se lee del `metrics.csv` que Lightning va guardando durante el entrenamiento (toma la mejor época por `mAP_50_95`). `RFDETRBase` no tiene método `.val()`.
- **Negativos:** la celda 2 NO toca los exports originales — reconstruye `data_rfdetr/` y `data_rtdetr/` desde cero en cada corrida. Cambiar `MODO_NEGATIVOS` o `NEG_RATIO_MAX` y reejecutar la celda 2 basta; no hace falta re-exportar de Roboflow.
- **Comparación justa:** `split_manifest.json` lista qué imagen va a train/valid. Pásaselo al compañero para que su YOLOv8 valide sobre exactamente las mismas imágenes que RT-DETR y RF-DETR.
- Si cambias de máquina, basta con copiar las carpetas `NuevasPlantas2.coco/`, `NuevasPlantas2.yolov8/` y los `.pth/.pt` — el notebook no depende de ninguna ruta absoluta hardcodeada.
