# Debugging a model inference pipeline

A small Python exercise for training, evaluating, and debugging an imbalanced binary classifier. It uses synthetic data, so the results demonstrate the evaluation method and do not estimate performance on any real dataset.

## Run on Windows

Clone or download this repository and open PowerShell in the project folder. With Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ml_exercise.py
.\.venv\Scripts\python.exe -m unittest -v
```

## Exercise

The logistic regression can return valid-looking probabilities even when the inputs use the wrong units. In `predict_positive`, temporarily replace the return line with this expression:

```python
return model.named_steps["logisticregression"].predict_proba(features)[:, 1]
```

Run the regression test by itself:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_ml_exercise.InferenceTests.test_inference_uses_fitted_scaler
```

It should fail. Explain why, then fix the function so it accepts raw features and uses the training-time transformation. Re-run both tests. The reference implementation is the original version of `predict_positive`.

## Evaluation

The stratified split reserves 300 rows for testing, including 34 positive cases. The scaler and model are fit on the training rows. Average precision (AP) summarizes how well the model ranks positive examples across thresholds. The baseline assigns the same score to every row, so its AP equals the positive fraction in the test set.

| Held-out result | Average precision |
| --- | ---: |
| Constant-score baseline | 0.113 |
| Fitted pipeline | 0.425 |
| Classifier with raw, unscaled inputs | 0.165 |

At the default threshold of 0.50, the pipeline finds 5 of 34 positives and raises 1 false positive. A better ranking can coexist with low recall at a specific threshold.

## Add your own evaluation

In `main`, calculate and print recall from the confusion matrix using `tp / (tp + fn)`. Run the script and explain what the result means in one sentence. For a follow-up exercise, reserve a separate validation split before choosing a threshold. Keep the test set for the final comparison.
