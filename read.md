
## 📊 Features Used

- `time_step`: time of measurement
- `u_in`: air flow into the lung (continuous)
- `u_out`: indicator if air is being exhaled
- `R`, `C`: lung resistance and compliance
- `breath_id`: ID to group 80 steps of a breath

## 🔧 Methods

### 1. Linear Regression
- Group data by `breath_id` (80 steps per breath)
- Z-score normalization on `u_in` and `pressure`
- Fit simple linear model using scikit-learn
- Predict pressure values per breath segment

### 2. LSTM with PyTorch
- Applied MinMax scaling on features
- Custom `Dataset` class to reshape time-series per breath
- Trained multi-layer LSTM model with MSE loss
- K-Fold cross-validation (GroupKFold) used for robustness
- Scheduler and GPU support included

## 📈 Results

| Model              | RMSE (Validation) | Notes                            |
|-------------------|-------------------|----------------------------------|
| Linear Regression | ~0.23             | Simple baseline, poor fit        |
| LSTM              | ~0.15             | Better fit on time-series data   |

*Note: Results may vary depending on split, epochs, and seed.*

## 📌 Key Learnings

- Breath-by-breath grouping is critical for meaningful training
- Simple models offer speed but miss temporal patterns
- LSTM improves performance significantly by modeling sequence
- Biomedical data needs careful normalization and scaling

## 🚀 Future Work

- Add attention mechanisms or Transformer models
- Compare with other models like CNN, GRU, or XGBoost
- Deploy prediction API or dashboard
- Integrate additional engineered features

## 👤 Author

**Leyin Qian**  
Master of Information Technology, UNSW  
Passionate about AI, time-series forecasting, and healthcare tech.  
[GitHub](https://github.com/your-github) | [LinkedIn](https://www.linkedin.com/in/leyin-qian/)

## 📦 Requirements

```bash
torch
numpy
pandas
matplotlib
scikit-learn
transformers