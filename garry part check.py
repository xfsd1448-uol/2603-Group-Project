# ============================================================
# Validation-stage Overfitting / Underfitting Check
#
# Purpose:
# Use the train-validation gap (rather than the test set) to assess
# whether each model shows signs of overfitting or underfitting.
#
# Why validation instead of test:
# The validation set is the appropriate reference during model
# development and model comparison. The test set should mainly be
# reserved for final performance reporting.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

# ------------------------------------------------------------
# 1. Validation-stage baseline model
# ------------------------------------------------------------
scaler_base_valcheck = StandardScaler()
X_train_base_valcheck = scaler_base_valcheck.fit_transform(X_train_raw)
X_val_base_valcheck   = scaler_base_valcheck.transform(X_val_raw)

baseline_valcheck_model = MLPRegressor(
    hidden_layer_sizes=(128, 32),
    activation='tanh',
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=800,
    early_stopping=False,
    random_state=26
)

baseline_valcheck_model.fit(X_train_base_valcheck, y_train)

baseline_train_pred_valcheck = baseline_valcheck_model.predict(X_train_base_valcheck)
baseline_val_pred_valcheck   = baseline_valcheck_model.predict(X_val_base_valcheck)

m1_train_mse_val = mean_squared_error(y_train, baseline_train_pred_valcheck)
m1_val_mse       = mean_squared_error(y_val, baseline_val_pred_valcheck)

# training loss curve (approximate MSE)
train_loss_curve_m1_val = np.array(baseline_valcheck_model.loss_curve_) * 2


# ------------------------------------------------------------
# 2. Validation-stage enhanced model
# ------------------------------------------------------------
best_K_valcheck = best_K

knn_valcheck = NearestNeighbors(n_neighbors=best_K_valcheck + 1)
knn_valcheck.fit(X_train_raw[['Latitude', 'Longitude']])

y_train_aligned_valcheck = y_train.loc[X_train_raw.index].to_numpy()

# training neighbours (remove self)
_, idx_train_valcheck = knn_valcheck.kneighbors(X_train_raw[['Latitude', 'Longitude']])
idx_train_valcheck = idx_train_valcheck[:, 1:best_K_valcheck + 1]

# validation neighbours (query against training only)
_, idx_val_valcheck = knn_valcheck.kneighbors(X_val_raw[['Latitude', 'Longitude']])
idx_val_valcheck = idx_val_valcheck[:, :best_K_valcheck]

train_prices_valcheck = y_train_aligned_valcheck[idx_train_valcheck]
val_prices_valcheck   = y_train_aligned_valcheck[idx_val_valcheck]

# neighbour mean price feature
neigh_mean_train_valcheck = train_prices_valcheck.mean(axis=1)
neigh_mean_valcheck       = val_prices_valcheck.mean(axis=1)

X_train_enh_valcheck_raw = X_train_raw.copy()
X_val_enh_valcheck_raw   = X_val_raw.copy()

X_train_enh_valcheck_raw["NeighbourPriceMean"] = neigh_mean_train_valcheck
X_val_enh_valcheck_raw["NeighbourPriceMean"]   = neigh_mean_valcheck

scaler_enh_valcheck = StandardScaler()
X_train_enh_valcheck = scaler_enh_valcheck.fit_transform(X_train_enh_valcheck_raw)
X_val_enh_valcheck   = scaler_enh_valcheck.transform(X_val_enh_valcheck_raw)

enhanced_valcheck_model = MLPRegressor(
    hidden_layer_sizes=(128, 32),
    activation='tanh',
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=800,
    early_stopping=False,
    random_state=26
)

enhanced_valcheck_model.fit(X_train_enh_valcheck, y_train)

enh_train_pred_valcheck = enhanced_valcheck_model.predict(X_train_enh_valcheck)
enh_val_pred_valcheck   = enhanced_valcheck_model.predict(X_val_enh_valcheck)

m2_train_mse_val = mean_squared_error(y_train, enh_train_pred_valcheck)
m2_val_mse       = mean_squared_error(y_val, enh_val_pred_valcheck)

# training loss curve (approximate MSE)
train_loss_curve_m2_val = np.array(enhanced_valcheck_model.loss_curve_) * 2


# ------------------------------------------------------------
# 3. Simple diagnosis based on validation / train ratio
# ------------------------------------------------------------
def diagnose_validation(model_name, train_mse, val_mse, threshold_ratio=1.5):

    ratio = val_mse / train_mse if train_mse > 0 else float('inf')

    print(f"\n{'='*60}")
    print(model_name)
    print(f"{'='*60}")
    print(f"Train MSE             : {train_mse:.4f}")
    print(f"Validation MSE        : {val_mse:.4f}")
    print(f"Validation/Train Ratio: {ratio:.3f}")

    if ratio > threshold_ratio:
        print("Diagnosis: possible overfitting (validation error notably exceeds training error)")
    elif ratio < 1.1:
        print("Diagnosis: small generalisation gap — possible underfitting if both errors remain relatively high compared with earlier configurations")
    else:
        print("Diagnosis: moderate generalisation gap / reasonable fit")

    print(f"{'='*60}")


diagnose_validation("Model 1 — Baseline (validation stage)", m1_train_mse_val, m1_val_mse)
diagnose_validation("Model 2 — Enhanced (validation stage)", m2_train_mse_val, m2_val_mse)


# ------------------------------------------------------------
# 4. Visualisation 1 — Train vs Validation MSE
# ------------------------------------------------------------
labels = ['Model 1\n(Baseline)', 'Model 2\n(Enhanced)']
train_mses_val = [m1_train_mse_val, m2_train_mse_val]
val_mses       = [m1_val_mse, m2_val_mse]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, train_mses_val, width, label='Train MSE', color='steelblue', alpha=0.85)
bars2 = ax.bar(x + width/2, val_mses,       width, label='Validation MSE', color='darkorange', alpha=0.85)

ax.set_ylabel('MSE')
ax.set_title('Train vs Validation MSE — Model 1 and Model 2')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bar, val in zip(bars1, train_mses_val):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)

for bar, val in zip(bars2, val_mses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 5. Visualisation 2 — Learning Curves
# ------------------------------------------------------------
print(f"Model 1 actual training epochs: {len(train_loss_curve_m1_val)}")
print(f"Model 2 actual training epochs: {len(train_loss_curve_m2_val)}")
print("Note: fewer epochs than max_iter indicate optimiser convergence before max_iter.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

global_min = min(
    train_loss_curve_m1_val.min(),
    train_loss_curve_m2_val.min(),
    m1_val_mse,
    m2_val_mse
)

global_max = max(
    train_loss_curve_m1_val.max(),
    train_loss_curve_m2_val.max(),
    m1_val_mse,
    m2_val_mse
)

for ax, lc_tr, final_train_mse, final_val_mse, title in zip(
    axes,
    [train_loss_curve_m1_val, train_loss_curve_m2_val],
    [m1_train_mse_val, m2_train_mse_val],
    [m1_val_mse, m2_val_mse],
    ['Model 1 — Baseline (validation stage)', 'Model 2 — Enhanced (validation stage)']
):

    epochs = range(1, len(lc_tr) + 1)

    ax.plot(epochs, lc_tr, label='Train MSE', color='steelblue')

    ax.axhline(
        y=final_train_mse,
        color='steelblue',
        linestyle=':',
        linewidth=1.2,
        label=f'Final Train MSE = {final_train_mse:.4f}'
    )

    ax.axhline(
        y=final_val_mse,
        color='darkorange',
        linestyle='--',
        linewidth=1.2,
        label=f'Final Validation MSE = {final_val_mse:.4f}'
    )

    ax.set_ylim(global_min * 0.9, global_max * 1.05)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Learning Curves — Validation-stage Generalisation Check', fontsize=13)
plt.tight_layout()
plt.show()