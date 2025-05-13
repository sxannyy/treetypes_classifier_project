import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

# Укажите путь к входному TIF-файлу и выходному PNG
# Например: input_tif_path = Path("/путь/к/файлу/classified_forest_rf.tif")

input_tif_path = Path("введите_путь_до_вашего/classified_forest_rf.tif")
output_png_path = Path("введите_путь_куда_сохранить/classified_map_rf.png")

label_map = {
    0: "siberian_cedar",
    1: "fir",
    2: "spruce",
    3: "dwarf_cedar",
    4: "not_covered",
    5: "pine",
    6: "aspen",
    7: "birch",
    8: "larch"
}

colors = [
    "#1a9850",  # siberian_cedar - сибирский кедр
    "#66bd63",  # fir - пихта
    "#a6d96a",  # spruce - ель
    "#d73027",  # dwarf_cedar - стланик
    "#f0f0f0",  # not_covered - нет данных
    "#3288bd",  # pine - сосна
    "#fee08b",  # aspen - осина
    "#fdae61",  # birch - берёза
    "#d53e4f"   # larch - лиственница
]

with rasterio.open(input_tif_path) as src:
    img = src.read(1)

img = np.where(img == -1, np.nan, img)

cmap = ListedColormap(colors)
norm = BoundaryNorm(range(10), cmap.N)

plt.figure(figsize=(12, 8))
im = plt.imshow(img, cmap=cmap, norm=norm)
plt.title("Карта классификации (Random Forest)", fontsize=14)
plt.axis("off")

class_labels = [label_map[i] for i in range(len(label_map))]
cbar = plt.colorbar(im, ticks=np.arange(0.5, 9.5, 1))
cbar.ax.set_yticklabels(class_labels)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(output_png_path, dpi=300)
plt.show()

print(f"Карта классификации сохранена в файл: {output_png_path}")