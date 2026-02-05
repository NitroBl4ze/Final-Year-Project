# Final Year Project - Real-Time Waste Classifier

This project uses a trained Keras model to classify waste categories.

## Files
- `app.py`: Streamlit live webcam app.
- `main.py`: OpenCV desktop webcam loop.
- `evaluate_model.py`: Batch evaluator that automatically reads all images in a dataset and reports performance.
- `keras_model.h5`: Trained model.
- `labels.txt`: Class labels.

## 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow keras opencv-python numpy pillow streamlit
```

## 2) Run live webcam app (web UI)
```bash
streamlit run app.py
```
Then open the URL shown in terminal (usually `http://localhost:8501`).

## 3) Run webcam app (desktop OpenCV)
```bash
python main.py
```
- Press `Esc` to stop.

## 4) Run automatic performance evaluation
## Automatic performance evaluation
Put your evaluation data in this structure:

```text
dataset/
  Cardboard/
    img1.jpg
    img2.jpg
  Glass/
  Metal/
  Paper/
  Plastic/
  Trash/
```

Run:
Then run:

```bash
python evaluate_model.py --dataset-dir dataset
```

It reports:
- Total samples and overall accuracy.
- Per-class precision, recall, F1-score, and support.
- Confusion matrix.

Optional custom paths:
You can override file paths:

```bash
python evaluate_model.py \
  --dataset-dir /path/to/test_data \
  --model-path keras_model.h5 \
  --labels-path labels.txt
```
