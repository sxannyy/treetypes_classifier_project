import json
import joblib
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

# Укажите пути к модели, списку бэндов, входному и выходному GeoTIFF

model_path = Path("rf_model.pkl")
band_file  = Path("band_order.txt")
tif_path   = Path("введите_путь_до_входного_файла/stack_with_ranges.tif")
out_path   = Path("введите_путь_до_выходного_файла/classified_forest_rf_smooth.tif")

CONF_TH = 0.35

model = joblib.load(model_path)
feature_order = json.loads(band_file.read_text())
class_codes = model.classes_

with rasterio.open(tif_path) as src:
    profile = src.profile
    band_names = src.descriptions
    h, w = src.height, src.width

    name2idx = {n: i for i, n in enumerate(band_names)}
    missing = [b for b in feature_order if b not in name2idx]
    if missing:
        raise ValueError(f"В GeoTIFF нет каналов: {', '.join(missing)}")

    idx_order = [name2idx[b] for b in feature_order]
    profile.update(dtype=rasterio.int16, count=1, compress="lzw", nodata=-1)

    n_cls = len(class_codes)
    votes = np.zeros((h, w, n_cls), dtype=np.uint16)

    overlap, tile_px = 16, 512
    tile_full = tile_px + 2 * overlap

    for row in range(0, h, tile_px):
        for col in range(0, w, tile_px):
            row_off = max(row - overlap, 0)
            col_off = max(col - overlap, 0)
            height = min(tile_full, h - row_off)
            width = min(tile_full, w - col_off)
            full = Window(col_off, row_off, width, height)

            arr = src.read(
                indexes=[i + 1 for i in idx_order],
                window=full,
                masked=True
            ).astype(np.float32)

            X = arr.data.reshape(len(idx_order), -1).T

            mask_full = arr.mask
            if np.isscalar(mask_full):
                valid = np.ones(X.shape[0], dtype=bool)
            else:
                valid = ~mask_full.reshape(len(idx_order), -1).any(axis=0)

            pred = np.full(X.shape[0], -1, np.int16)
            if valid.any():
                proba = model.predict_proba(X[valid])
                conf = proba.max(axis=1)
                cls = class_codes[proba.argmax(axis=1)]
                keep = conf > CONF_TH
                pred_idx = np.where(valid)[0][keep]
                pred[pred_idx] = cls[keep]

            pred_2d = pred.reshape(arr.shape[1:])

            inner = Window(col, row,
                           min(tile_px, w - col),
                           min(tile_px, h - row))
            r0 = inner.row_off - full.row_off
            c0 = inner.col_off - full.col_off
            r1, c1 = r0 + inner.height, c0 + inner.width

            inner_pred = pred_2d[r0:r1, c0:c1]
            mask_valid = inner_pred != -1
            rr = np.arange(inner.height) + row
            cc = np.arange(inner.width) + col

            for k, code in enumerate(class_codes):
                mask_k = mask_valid & (inner_pred == code)
                if mask_k.any():
                    votes[np.ix_(rr, cc, [k])] += mask_k[..., None].astype(np.uint16)

    majority_idx = votes.argmax(axis=2)
    final_map = class_codes[majority_idx].astype(np.int16)
    final_map[votes.sum(axis=2) == 0] = -1

with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(final_map, 1)

print("Классификация завершена:", out_path)