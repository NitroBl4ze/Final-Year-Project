# Final Year Project - Real-Time Waste Classifier

This project uses a trained Keras model to classify waste categories.

## Files
- `app.py`: Streamlit live webcam app.
- `main.py`: OpenCV desktop webcam loop.
- `evaluate_model.py`: Batch evaluator that automatically reads all images in a dataset and reports performance.
- `keras_model.h5`: Trained model.
- `labels.txt`: Class labels.

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

Then run:

```bash
python evaluate_model.py --dataset-dir dataset
```

It reports:
- Total samples and overall accuracy.
- Per-class precision, recall, F1-score, and support.
- Confusion matrix.

You can override file paths:

```bash
python evaluate_model.py \
  --dataset-dir /path/to/test_data \
  --model-path keras_model.h5 \
  --labels-path labels.txt
```
