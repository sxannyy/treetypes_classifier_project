import rasterio
import numpy as np
from pathlib import Path

# Укажите путь к входному и выходному файлу
# Например: input_path = Path("/путь/к/вашему/файлу/merged_stack.tif")

input_path  = Path("введите_путь_до_вашего_входного_файла/merged_stack.tif")
output_path = Path("введите_путь_до_вашего_выходного_файла/stack_with_ranges.tif")

index_pairs = [
    ("NDVI_summer",  "NDVI_winter",  "NDVI_range"),
    ("EVI_summer",   "EVI_winter",   "EVI_range"),
    ("GNDVI_summer", "GNDVI_winter", "GNDVI_range"),
    ("NDMI_summer",  "NDMI_winter",  "NDMI_range"),
    ("NBR_summer",   "NBR_winter",   "NBR_range"),
]

EPS = 1e-5

with rasterio.open(input_path) as src:
    profile       = src.profile
    descriptions  = list(src.descriptions)
    band_map      = {name: i + 1 for i, name in enumerate(descriptions)}
    data          = [src.read(i + 1) for i in range(src.count)]

    for summer_name, winter_name, range_name in index_pairs:
        if summer_name not in band_map or winter_name not in band_map:
            print(f"Внимание! Пропущена пара: {summer_name}, {winter_name}")
            continue

        b_summer = src.read(band_map[summer_name]).astype(np.float32)
        b_winter = src.read(band_map[winter_name]).astype(np.float32)

        data.append((b_summer - b_winter).astype(np.float32))
        descriptions.append(range_name)

        ratio_name = range_name.replace("_range", "_ratio")
        data.append((b_summer / (b_winter + EPS)).astype(np.float32))
        descriptions.append(ratio_name)

    profile.update(count=len(data), dtype=rasterio.float32, compress="lzw")

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, arr in enumerate(data):
            dst.write(arr, i + 1)
        dst.descriptions = tuple(descriptions)

print("Создан файл:", output_path)
print("Каналы:", descriptions)