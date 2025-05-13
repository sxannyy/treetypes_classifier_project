import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# Укажите пути к CSV, модели, порядку бэндов, карте меток и файлу важности признаков

csv_path = Path("введите_путь_до/training_rf_samples.csv")
model_path = Path("rf_model.pkl")
band_file = Path("band_order.txt")
label_map_path = Path("введите_путь_до/label_map.json")
feature_importance_path = Path("rf_feature_importances.json")

EPS = 1e-5

df = pd.read_csv(csv_path)

if "type" in df.columns:
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["type"])
    label_map = dict(zip(le.transform(le.classes_), le.classes_))
    label_map_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2))
else:
    loaded = json.loads(label_map_path.read_text())
    if isinstance(next(iter(loaded.keys())), str):
        label_map = {v: k for k, v in loaded.items()}
    else:
        label_map = loaded

for idx in ["NDVI", "EVI", "GNDVI", "NDMI", "NBR"]:
    if f"{idx}_summer" in df.columns and f"{idx}_winter" in df.columns:
        df[f"{idx}_range"] = df[f"{idx}_summer"] - df[f"{idx}_winter"]
        df[f"{idx}_ratio"] = df[f"{idx}_summer"] / (df[f"{idx}_winter"] + EPS)

exclude_cols = {"label", "id", "type", "system:index", ".geo"}
feature_cols = [c for c in df.columns if c not in exclude_cols]
df[feature_cols] = df[feature_cols].astype(np.float32)

X = df[feature_cols].values
y = df["label"].astype(np.int16).values

rare_threshold = 100
class_counts = pd.Series(y).value_counts()
rare_classes = class_counts[class_counts < rare_threshold].index.tolist()

if rare_classes:
    sampling_strategy = {cls: rare_threshold * 2 for cls in rare_classes}
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    sample_idx_attr = "sample_indices_" if hasattr(ros, "sample_indices_") else "_sample_indices"
    sample_indices  = getattr(ros, sample_idx_attr)
else:
    X_res, y_res = X, y
    sample_indices  = np.arange(len(y))

if "id" in df.columns:
    orig_groups = df["id"].values
    groups_res = orig_groups[sample_indices]
else:
    groups_res = np.arange(len(y_res))

gkf = GroupKFold(n_splits=5)
val_scores = []
num_classes = len(label_map)
all_cm = np.zeros((num_classes, num_classes), dtype=int)

for fold, (tr, vl) in enumerate(gkf.split(X_res, y_res, groups_res), start=1):
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X_res[tr], y_res[tr])
    preds = model.predict(X_res[vl])
    acc   = (preds == y_res[vl]).mean()
    val_scores.append(acc)

    all_cm += confusion_matrix(y_res[vl], preds, labels=list(range(num_classes)))

    print(f"Fold {fold} accuracy: {acc:.4f}")

print(f"\nСредняя точность (5-fold CV): {np.mean(val_scores):.4f}")

final_model = RandomForestClassifier(
    n_estimators = 500,
    max_depth = None,
    max_features = "sqrt",
    min_samples_leaf = 3,
    n_jobs = -1,
    random_state = 42,
    class_weight = "balanced_subsample"
)
final_model.fit(X_res, y_res)

fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False)
feature_importance_path.write_text(
    json.dumps(dict(zip(fi["feature"], fi["importance"])), indent=2)
)

joblib.dump(final_model, model_path)
band_file.write_text(json.dumps(feature_cols))

plt.figure(figsize=(10, 6))
plt.barh(fi["feature"], fi["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(
    all_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[label_map[i] for i in range(num_classes)],
    yticklabels=[label_map[i] for i in range(num_classes)],
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (5-fold CV)")
plt.tight_layout()
plt.show()

print("\nМодель, порядок бэндов и важности сохранены.")