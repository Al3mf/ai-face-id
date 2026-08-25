# AI Face ID

Local face authentication with Haar Cascade detection and LBPH recognition (OpenCV).

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

1. Create a new profile and capture about 30 face images. The program asks for the profile name. Press SPACE to save a face and ESC to finish:

```powershell
python face_auth.py capture --count 30
```

2. Train the model:

```powershell
python face_auth.py train
```

3. Live authentication (green = granted, red = denied; ESC to exit):

```powershell
python face_auth.py recognize
```

## Notes

- Access is granted when LBPH distance is below `70`.
- Dataset images go to `dataset/<profile>/`. The model is saved as `model/face_model.yml` and `model/label_map.json`.
- Webcam capture uses OpenCV (`CAP_DSHOW` on Windows).

## Contributors

- [Alejandro Melo (@Al3mf)](https://github.com/Al3mf)
- [Montse Sanchez (@sanchezmontse)](https://github.com/sanchezmontse)
- [Rodolfo Monreal (@Rodo3118)](https://github.com/Rodo3118)
- [Sergio Barajas (@barajasequationz)](https://github.com/barajasequationz)

## License

This project is licensed under the [MIT License](LICENSE).
