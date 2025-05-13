import rasterio
from rasterio.merge import merge
from pathlib import Path

# Укажите путь к папке с тайлами и выходному файлу

tile_dir = Path("введите_путь_к_директории_с_тайлами")
output = tile_dir / "merged_stack.tif"

tif_files = list(tile_dir.glob("stack_seasonal_climate-*.tif"))
if not tif_files:
    raise FileNotFoundError("Файлы с шаблоном 'stack_seasonal_climate-*.tif' не найдены.")

src_files = [rasterio.open(f) for f in tif_files]
mosaic, transform = merge(src_files, method="max")

meta = src_files[0].meta.copy()
meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": transform,
    "compress": "lzw"
})

descriptions = src_files[0].descriptions
print("Сохраняемые описания:", descriptions)

with rasterio.open(output, "w", **meta) as dest:
    dest.write(mosaic)
    if all(descriptions):
        dest.descriptions = descriptions

for src in src_files:
    src.close()

print("Склеено с сохранением descriptions ->", output)