
import os, time, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------- USER CONFIG ----------
DATA_CSV = "hemo_dataset.csv"
SEQ_LEN = 50
MAX_WINDOWS = 60000
BATCH_SIZE = 256
EPOCHS = 60
LR = 5e-4
OUTDIR = "./gru_target96"
os.makedirs(OUTDIR, exist_ok=True)

GRU_UNITS = 64
DENSE_UNITS = 96
DROPOUT = 0.65
L2 = 5e-3
GAUSSIAN_NOISE_STD = 0.12
LABEL_NOISE = 0.02   # 2% label flips
USE_MIXED_PRECISION = True

# Quick light grid for dropout/L2 (fast trials)
RUN_LIGHT_FT = True
FT_TRIALS = [
    {"dropout":0.60, "l2":3e-3},
    {"dropout":0.65, "l2":5e-3},
    {"dropout":0.70, "l2":8e-3},
]
# ----------------------------------

FEATURES = ['ART','ECG','MAP','HR','SpO2']
LABEL = 'hypotension_label'

# reproducibility
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# environment prep
tf.keras.backend.clear_session(); gc.collect()
print("TensorFlow version:", tf.__version__, "GPUs:", tf.config.list_physical_devices('GPU'))
if USE_MIXED_PRECISION:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled (policy:", mixed_precision.global_policy(), ")")
    except Exception as e:
        print("Mixed precision not enabled:", e)

# ---------------- load & validate CSV ----------------
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"Dataset not found at: {DATA_CSV}")

df = pd.read_csv(DATA_CSV)
missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")
print("CSV loaded. Rows:", len(df))

# ---------------- ENHANCED PREPROCESSING ----------------
print("\n" + "="*50)
print("ENHANCED PREPROCESSING PIPELINE")
print("="*50)

# 1. Data Quality Checks
print("\n1. DATA QUALITY CHECKS:")
print(f"   Original dataset shape: {df.shape}")
print(f"   Missing values per column:")
for col in FEATURES + [LABEL]:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        print(f"     {col}: {missing_count} missing values ({missing_count/len(df)*100:.2f}%)")

# Handle missing values
df_clean = df.copy()
for col in FEATURES:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
        print(f"   Filled missing values in {col} with median")

# 2. Outlier Detection and Treatment
print("\n2. OUTLIER HANDLING:")
for col in FEATURES:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    print(f"   {col}: {outliers} outliers ({outliers/len(df_clean)*100:.2f}%)")

    # Cap outliers
    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

# 3. Feature Distribution Analysis
print("\n3. FEATURE DISTRIBUTION ANALYSIS:")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(FEATURES):
    axes[i].hist(df_clean[col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[i].set_title(f'{col} Distribution')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

    # Add statistics
    mean_val = df_clean[col].mean()
    std_val = df_clean[col].std()
    axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1 STD')
    axes[i].axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
    axes[i].legend()

# Remove empty subplot
axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Label Distribution
print("\n4. LABEL DISTRIBUTION:")
label_counts = df_clean[LABEL].value_counts()
print(f"   Class 0: {label_counts[0]} samples ({label_counts[0]/len(df_clean)*100:.2f}%)")
print(f"   Class 1: {label_counts[1]} samples ({label_counts[1]/len(df_clean)*100:.2f}%)")

plt.figure(figsize=(8, 6))
plt.bar(['Class 0', 'Class 1'], [label_counts[0], label_counts[1]], color=['lightblue', 'lightcoral'])
plt.title('Class Distribution')
plt.ylabel('Number of Samples')
for i, v in enumerate([label_counts[0], label_counts[1]]):
    plt.text(i, v + max([label_counts[0], label_counts[1]])*0.01, str(v), ha='center', va='bottom')
plt.savefig(os.path.join(OUTDIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Correlation Analysis
print("\n5. CORRELATION ANALYSIS:")
correlation_matrix = df_clean[FEATURES + [LABEL]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print("   Correlation with target:")
for feature in FEATURES:
    corr = df_clean[feature].corr(df_clean[LABEL])
    print(f"     {feature}: {corr:.3f}")

# 6. Standard Scaling
print("\n6. APPLYING STANDARD SCALING...")
scaler = StandardScaler()
scaled_all = scaler.fit_transform(df_clean[FEATURES])
joblib.dump(scaler, os.path.join(OUTDIR, "scaler.save"))

# 7. Time-series Sequence Creation
print("\n7. CREATING TIME-SERIES SEQUENCES...")
total_windows = max(0, len(df_clean) - SEQ_LEN)
take = min(total_windows, MAX_WINDOWS)
print(f"   Total windows available: {total_windows}, sampling: {take}")

labels_all = df_clean[LABEL].values[SEQ_LEN-1:SEQ_LEN-1 + total_windows]

# Balanced sampling
pos = np.where(labels_all == 1)[0]
neg = np.where(labels_all == 0)[0]
half = take // 2
if len(pos) >= half:
    s_pos = np.random.choice(pos, half, replace=False)
else:
    s_pos = np.random.choice(pos, half, replace=True) if len(pos) > 0 else np.array([], dtype=int)
s_neg = np.random.choice(neg, take - len(s_pos), replace=False)
sel = np.concatenate([s_pos, s_neg])
np.random.shuffle(sel)

# Build sequence windows
n_samples = len(sel)
n_features = len(FEATURES)
X = np.empty((n_samples, SEQ_LEN, n_features), dtype=np.float32)
y = np.empty((n_samples,), dtype=np.float32)
for i, idx in enumerate(sel):
    X[i] = scaled_all[idx:idx+SEQ_LEN]
    y[i] = labels_all[idx]

# Optional label noise
if LABEL_NOISE and LABEL_NOISE > 0:
    n_flip = int(len(y) * LABEL_NOISE)
    if n_flip > 0:
        flip_idx = np.random.choice(len(y), n_flip, replace=False)
        y[flip_idx] = 1 - y[flip_idx]
        print(f"   Applied label noise: flipped {n_flip} labels")

print("   Prepared data shapes:", X.shape, y.shape)

# 8. Data Splits
print("\n8. DATA SPLITTING...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.27, random_state=SEED, stratify=y)
val_frac = 0.12 / 0.27
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_frac), random_state=SEED, stratify=y_temp)
print("   Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

# Class weights
classes = np.unique(y_train)
cw_vals = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, cw_vals)}
print("   Class weights:", class_weights)

# tf.data pipelines
AUTOTUNE = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(20000, seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1024)

# ---------------- model builder ----------------
def build_gru(seq_len, n_features, gru_units=GRU_UNITS, dense_units=DENSE_UNITS, dropout=DROPOUT, l2=L2):
    inp = keras.Input(shape=(seq_len, n_features))
    x = layers.GaussianNoise(GAUSSIAN_NOISE_STD)(inp)
    x = layers.Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.GRU(gru_units, return_sequences=False,
                   kernel_regularizer=regularizers.l2(l2),
                   recurrent_regularizer=regularizers.l2(l2),
                   recurrent_dropout=0.25)(x)   # recurrent dropout reduces capacity a bit
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout * 0.6)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32', kernel_regularizer=regularizers.l2(l2))(x)
    return keras.Model(inp, out)

def compile_model(m, lr=LR):
    opt = keras.optimizers.Adam(learning_rate=lr)
    m.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC(name='auc')])

# quick light fine-tune sweep to choose dropout/L2 (short runs)
best_cfg = {"dropout":DROPOUT, "l2":L2}
if RUN_LIGHT_FT:
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    best_auc = -1.0
    ft_results = []
    for cfg in FT_TRIALS:
        print("Trial:", cfg)
        m = build_gru(SEQ_LEN, n_features, gru_units=GRU_UNITS, dense_units=DENSE_UNITS,
                      dropout=cfg["dropout"], l2=cfg["l2"])
        compile_model(m, lr=LR)
        # tiny early stopping, short epochs
        history = m.fit(train_ds, validation_data=val_ds, epochs=6,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=2, restore_best_weights=True)],
                        class_weight=class_weights, verbose=0)
        val_auc = max(history.history.get('val_auc', [0.0]))
        ft_results.append({'config': cfg, 'val_auc': val_auc})
        print(f" -> val_auc {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_cfg = cfg.copy()
    print("Selected regularization cfg:", best_cfg, "val_auc:", best_auc)
    DROPOUT = best_cfg["dropout"]
    L2 = best_cfg["l2"]

# build and compile final model
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)
model = build_gru(SEQ_LEN, n_features, gru_units=GRU_UNITS, dense_units=DENSE_UNITS, dropout=DROPOUT, l2=L2)
compile_model(model)
model.summary()

# callbacks & training
ckpt = os.path.join(OUTDIR, "gru_target96_best.h5")
callbacks = [
    keras.callbacks.ModelCheckpoint(ckpt, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True, verbose=1)
]

print("Starting full training...")
start_time = time.time()
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks,
                 class_weight=class_weights, verbose=1)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# ---------------- ENHANCED EVALUATION & PLOTTING ----------------
print("\n" + "="*50)
print("COMPREHENSIVE EVALUATION & VISUALIZATION")
print("="*50)

if os.path.exists(ckpt):
    best = keras.models.load_model(ckpt)
    print("Loaded best model from checkpoint")
else:
    best = model
    print("Using final model (no checkpoint found)")

# Predictions
print("\nGenerating predictions...")
preds = best.predict(test_ds).ravel()
pred_cls = (preds >= 0.5).astype(int)

# Comprehensive Metrics
test_auc = roc_auc_score(y_test, preds)
test_acc = accuracy_score(y_test, pred_cls)
test_report = classification_report(y_test, pred_cls, digits=4)
test_cm = confusion_matrix(y_test, pred_cls)

print(f"\n{' RESULTS ':=^60}")
print(f"TEST AUC:  {test_auc:.4f}")
print(f"TEST ACC:  {test_acc:.4f}")
print(f"\nCLASSIFICATION REPORT:\n{test_report}")
print(f"CONFUSION MATRIX:\n{test_cm}")

# 1. Training History Plots
print("\nGenerating training history plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Loss
axes[0,0].plot(hist.history['loss'], label='Training Loss', linewidth=2)
axes[0,0].plot(hist.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0,0].set_title('Model Loss')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Accuracy
axes[0,1].plot(hist.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0,1].plot(hist.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0,1].set_title('Model Accuracy')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Accuracy')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# AUC
axes[1,0].plot(hist.history['auc'], label='Training AUC', linewidth=2)
axes[1,0].plot(hist.history['val_auc'], label='Validation AUC', linewidth=2)
axes[1,0].set_title('Model AUC')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('AUC')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Learning Rate
if 'lr' in hist.history:
    axes[1,1].plot(hist.history['lr'], label='Learning Rate', linewidth=2, color='purple')
    axes[1,1].set_title('Learning Rate')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].set_yscale('log')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
else:
    axes[1,1].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. ROC Curve
from sklearn.metrics import roc_curve, precision_recall_curve

fpr, tpr, thresholds = roc_curve(y_test, preds)
roc_auc = roc_auc_score(y_test, preds)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTDIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, preds)
avg_precision = np.mean(precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTDIR, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Prediction Distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(preds[y_test == 0], bins=50, alpha=0.7, color='blue', label='Class 0', density=True)
plt.hist(preds[y_test == 1], bins=50, alpha=0.7, color='red', label='Class 1', density=True)
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Prediction Distribution by True Class')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Calibration plot
bin_means = np.linspace(0, 1, 11)
bin_centers = (bin_means[:-1] + bin_means[1:]) / 2
bin_true = []
for i in range(len(bin_means)-1):
    mask = (preds >= bin_means[i]) & (preds < bin_means[i+1])
    if mask.any():
        bin_true.append(np.mean(y_test[mask]))
    else:
        bin_true.append(0)

plt.plot(bin_centers, bin_true, 's-', label='Model Calibration')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('True Fraction Positive')
plt.title('Calibration Plot')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Results Summary
results_summary = {
    'test_auc': test_auc,
    'test_accuracy': test_acc,
    'training_time_seconds': training_time,
    'best_epoch': len(hist.history['loss']),
    'final_training_loss': hist.history['loss'][-1],
    'final_validation_loss': hist.history['val_loss'][-1],
    'final_training_auc': hist.history['auc'][-1],
    'final_validation_auc': hist.history['val_auc'][-1],
    'best_hyperparameters': best_cfg,
    'class_weights': class_weights,
    'dataset_info': {
        'total_samples': len(df_clean),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'sequence_length': SEQ_LEN,
        'features': FEATURES
    }
}

# Save results summary
results_df = pd.DataFrame([results_summary])
results_df.to_csv(os.path.join(OUTDIR, 'results_summary.csv'), index=False)

# Print comprehensive results
print(f"\n{' DETAILED RESULTS SUMMARY ':=^60}")
print(f"Model Performance:")
print(f"  AUC Score:        {test_auc:.4f}")
print(f"  Accuracy:         {test_acc:.4f}")
print(f"  Training Time:    {training_time:.2f} seconds")
print(f"  Best Epoch:       {len(hist.history['loss'])}")
print(f"\nDataset Info:")
print(f"  Total Samples:    {len(df_clean)}")
print(f"  Train Set:        {len(X_train)}")
print(f"  Validation Set:   {len(X_val)}")
print(f"  Test Set:         {len(X_test)}")
print(f"  Sequence Length:  {SEQ_LEN}")
print(f"  Features:         {', '.join(FEATURES)}")
print(f"\nBest Hyperparameters:")
print(f"  Dropout:          {best_cfg['dropout']}")
print(f"  L2 Regularization:{best_cfg['l2']}")
print(f"  Class Weights:    {class_weights}")

# Save final artifacts
try:
    best.save(os.path.join(OUTDIR, "gru_target96_final.h5"))
    joblib.dump(scaler, os.path.join(OUTDIR, "scaler.save"))

    # Save training history for plotting
    history_df = pd.DataFrame(hist.history)
    history_df.to_csv(os.path.join(OUTDIR, "training_history.csv"), index=False)

    print(f"\nSaved all artifacts to: {OUTDIR}")
    print("  - Model weights (.h5)")
    print("  - Scaler (.save)")
    print("  - Training history (.csv)")
    print("  - Results summary (.csv)")
    print("  - Multiple visualization plots (.png)")
except Exception as e:
    print("Warning: could not save some artifacts:", e)

print(f"\n{' ANALYSIS COMPLETE ':=^60}")