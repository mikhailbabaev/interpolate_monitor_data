import os
import numpy as np
import cv2
from time import perf_counter
from scipy.spatial import cKDTree


def read_file(filename):
    """Чтение из файла npu"""
    data = np.load(filename, allow_pickle=False)
    return data


def plot_grid(data, input_filename, width=500, height=500):
    """Строим грид по массиву"""
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # Приведение координат к масштабу изображения
    x_scaled = ((x - x.min()) / (x.max() - x.min()) * (width - 1)).astype(int)
    y_scaled = ((y - y.min()) / (y.max() - y.min()) * (height - 1)).astype(int)
    # Нормализация z для отображения в диапазоне RGB
    z_normalized = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)

    heatmap = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(x_scaled)):
        heatmap[y_scaled[i], x_scaled[i]] = z_normalized[i]

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    output_filename = os.path.splitext(input_filename)[0] + '_plot.png'
    cv2.imwrite(output_filename, heatmap_colored)
    return x, y, z


def interpolate_data(source_x, source_y, source_z, target_x, target_y, k=5):
    """
    Интерполирует значения из source (x, y, z) на координаты target (x, y)
    """
    # Построение KDTree для поиска ближайших соседей
    tree = cKDTree(np.c_[source_x, source_y])

    # Поиск ближайших k соседей для целевых точек
    distances, indices = tree.query(np.c_[target_x, target_y], k=k)

    weights = 1 / (distances + 1e-10)

    # Средневзвешенное значение z для соседей (IDW)
    interpolated_z = (
        np.sum(source_z[indices] * weights, axis=1) /
        np.sum(weights, axis=1)
    )

    return interpolated_z


# Файлы данных
file1 = 'data1.npy'
file2 = 'data2.npy'

# Чтение данных из npu
start_time = perf_counter()
data_1 = read_file(file1)
data_2 = read_file(file2)
print(f'Чтение файлов: {perf_counter() - start_time:.4f} секунд')

# Нормализация и построение карт
start_time = perf_counter()
x1, y1, z1 = plot_grid(data_1, file1)
x2, y2, z2 = plot_grid(data_2, file2)
print(f'Нормализация и построение карт: '
      f'{perf_counter() - start_time:.4f} секунд')

# Интерполяция data_1 на координаты data_2
start_time = perf_counter()
interpolated_z1_on_data2 = interpolate_data(x1, y1, z1, x2, y2, k=15)
print(f'Интерполяция: {perf_counter() - start_time:.4f} секунд')

# Сравнение интерполированных значений и исходных
start_time = perf_counter()
difference = z2 - interpolated_z1_on_data2
print(f'Сравнение данных: {perf_counter() - start_time:.4f} секунд')

# Сохранение результата в новый файл png
start_time = perf_counter()
output_difference_file = 'difference_map.png'
plot_grid(np.c_[x2, y2, difference], output_difference_file)
print(f'Сохранение результата: {perf_counter() - start_time:.4f} секунд')


# Сохранение результата в новый файл txt и npy
start_time = perf_counter()
output_interpolated_file = 'interpolated_data1_on_data2.txt'
interpolated_data = np.c_[x2, y2, interpolated_z1_on_data2]
np.save('interpolated.npy', interpolated_data)
np.savetxt(output_interpolated_file, interpolated_data, delimiter=',',
           fmt='%.6f', header='x,y,interpolated_z')
print(f'Интерполированные данные сохранены в файл: {output_interpolated_file}'
      f' за {perf_counter() - start_time:.4f} секунд')
