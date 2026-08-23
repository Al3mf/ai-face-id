# AI Face ID

Local port of the Colab face-authentication notebook (Haar detection + LBPH recognition).

Original notebook: [Colab](https://colab.research.google.com/drive/1a6IrMLw_6yxn2Viofm7AEoXEs9VsnyeL?usp=sharing)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `cv2.face` is missing, uninstall regular OpenCV and keep only the contrib build:

```powershell
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
pip install opencv-contrib-python==4.10.0.84
```

## Run

1. Capture ~30 face images (SPACE to save, ESC to finish):

```powershell
python face_auth.py capture --user "Alejandro Melo" --count 30
```

2. Train the model:

```powershell
python face_auth.py train
```

3. Live authentication (green = granted, red = denied; ESC to exit):

```powershell
python face_auth.py recognize
```

A Jupyter version of the same flow is in `face_authentication.ipynb`.

## Notes

- Access is granted when LBPH distance is below `70` (same threshold as the Colab notebook).
- Dataset images go to `dataset/<user>/`. The model is saved as `model/face_model.yml` and `model/label_map.json`.
- Webcam capture uses OpenCV (`CAP_DSHOW` on Windows) instead of Google Colab JavaScript.
