"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)

def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    if vectors.ndim == 1:  # один вектор (n,) -> (1, n, 1)
        vectors = vectors.reshape(1, -1, 1)
    elif vectors.ndim == 2:  # набор векторов (p, n) -> (p, n, 1)
        vectors = vectors[:, :, np.newaxis]

    # (p, n, n) @ (p, n, 1) -> (p, n, 1), затем сумма по p -> (n, 1)
    return np.matmul(matrices, vectors).sum(axis=0)

def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > data.threshold).astype(int)

def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in matrix]

def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(matrix[:, column]).tolist() for column in range(matrix.shape[1])]

def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    generator = np.random.default_rng(data.seed)
    matrix = generator.normal(loc=data.mean, scale=data.std, size=(data.rows, data.columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )

def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    dtype = np.result_type(data.first, data.second)
    board = np.full((data.rows, data.columns), data.first, dtype=dtype)
    row_index, column_index = np.indices((data.rows, data.columns))
    board[(row_index + column_index) % 2 == 1] = data.second
    return board

def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((data.image_height, data.image_width, 3), dtype=np.uint8)
    image[:, :] = np.asarray(data.background_color, dtype=np.uint8)

    width = min(data.width, data.image_width)
    height = min(data.height, data.image_height)
    top = (data.image_height - height) // 2
    left = (data.image_width - width) // 2
    image[top:top + height, left:left + width] = np.asarray(data.shape_color, dtype=np.uint8)
    return image

def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((data.image_height, data.image_width, 3), dtype=np.uint8)
    image[:, :] = np.asarray(data.background_color, dtype=np.uint8)

    x0 = (data.image_width - 1) / 2
    y0 = (data.image_height - 1) / 2
    y_coordinates, x_coordinates = np.ogrid[:data.image_height, :data.image_width]
    inside = (
                     (x_coordinates - x0) ** 2 / data.semi_axis_x ** 2
                     + (y_coordinates - y0) ** 2 / data.semi_axis_y ** 2
             ) <= 1
    image[inside] = np.asarray(data.shape_color, dtype=np.uint8)
    return image

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(data.values, dtype=float)
    window = data.window

    if values.size >= 3:
        previous, current, following = values[:-2], values[1:-1], values[2:]
        local_maxima_indices = np.nonzero((current > previous) & (current > following))[0] + 1
        local_minima_indices = np.nonzero((current < previous) & (current < following))[0] + 1
    else:  # у первого и последнего элемента соседей не хватает
        empty = np.array([], dtype=int)
        local_maxima_indices, local_minima_indices = empty, empty.copy()

    if 1 <= window <= values.size:
        kernel = np.ones(window) / window
        moving_average = np.convolve(values, kernel, mode="valid")  # длина n - p + 1
    else:
        moving_average = np.array([])

    return TimeSeriesStatistics(
        mean=values.mean(),
        variance=values.var(),
        std=values.std(),
        local_maxima_indices=local_maxima_indices,
        local_minima_indices=local_minima_indices,
        moving_average=moving_average,
    )

def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(data.labels, dtype=int).ravel()
    class_count = data.class_count
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0

    encoded = np.zeros((labels.size, class_count), dtype=int)
    valid = (labels >= 0) & (labels < class_count)  # защита от меток вне диапазона
    encoded[np.arange(labels.size)[valid], labels[valid]] = 1
    return encoded

# Тесты: простой прогон функций на разных значениях, включая граничные случаи.
np.set_printoptions(precision=4, suppress=True, linewidth=120)

print("=== Задача 1. sum_prod ===")
matrices = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])   # 2 матрицы 2x2
vectors = np.array([[1, 0], [0, 1]])                        # 2 вектора (2, 1)
print("shape:", sum_prod(MatrixVectorBatchInput(matrices, vectors)).shape)
print(sum_prod(MatrixVectorBatchInput(matrices, vectors)))  # [1,3] + [6,8] = [7,11]
print(sum_prod(MatrixVectorBatchInput(np.eye(3), [1, 2, 3])).ravel())  # одна матрица

print("\n=== Задача 2. binarize ===")
sample = np.array([[0.1, 0.5, 0.9], [1.0, 0.0, 0.5]])
print(binarize(BinarizeInput(sample, 0.5)))                 # порог не включается
print(binarize(BinarizeInput(sample, 0.0)))
print(binarize(BinarizeInput(np.array([[1, 2, 3], [4, 5, 6]]), 3)))
print("исходная матрица не изменилась:", sample.tolist())

print("\n=== Задача 3. unique_rows / unique_columns ===")
sample = np.array([[1, 2, 2, 3], [3, 3, 3, 1], [0, 1, 0, 1]])
print("rows:   ", unique_rows(MatrixInput(sample)))
print("columns:", unique_columns(MatrixInput(sample)))

print("\n=== Задача 4. matrix_statistics ===")
stats = matrix_statistics(RandomMatrixInput(rows=4, columns=5, mean=10.0, std=2.0, seed=42))
print("matrix:\n", stats.matrix)
print("row_means:      ", stats.row_means)
print("column_means:   ", stats.column_means)
print("row_variances:  ", stats.row_variances)
print("column_variances:", stats.column_variances)
print("проверка row_means:", np.allclose(stats.row_means, stats.matrix.mean(axis=1)))

print("\n=== Задача 5. chess ===")
print(chess(ChessInput(rows=3, columns=4, first=0, second=1)))
print(chess(ChessInput(rows=1, columns=1, first=7, second=9)))
print(chess(ChessInput(rows=4, columns=4, first=0.5, second=-0.5)))

print("\n=== Задача 6. draw_rectangle / draw_ellipse ===")
rectangle = draw_rectangle(RectangleInput(width=4, height=2, image_height=6, image_width=8,
                                          shape_color=(255, 0, 0), background_color=(0, 0, 0)))
print("shape:", rectangle.shape, "dtype:", rectangle.dtype)
print("красный канал:\n", rectangle[:, :, 0])
ellipse = draw_ellipse(EllipseInput(semi_axis_x=3, semi_axis_y=2, image_height=9,
                                    image_width=9, shape_color=(0, 255, 0),
                                    background_color=(255, 255, 255)))
print("shape:", ellipse.shape)
print("красный канал (0 — эллипс, 255 — фон):\n", ellipse[:, :, 0])
print("центр окрашен:", (ellipse[4, 4] == (0, 255, 0)).all(),
      "| угол фоновый:", (ellipse[0, 0] == (255, 255, 255)).all(),
      "| площадь:", int((ellipse[:, :, 0] == 0).sum()), "≈ πab =", round(np.pi * 3 * 2, 2))

print("\n=== Задача 7. analyze_time_series ===")
series = np.array([1.0, 3.0, 2.0, 5.0, 5.0, 1.0, 4.0])
result = analyze_time_series(TimeSeriesInput(series, window=3))
print("mean:", result.mean, "| variance:", result.variance, "| std:", result.std)
print("local_maxima_indices:", result.local_maxima_indices)   # плато 5,5 не максимум
print("local_minima_indices:", result.local_minima_indices)
print("moving_average:", result.moving_average, "| длина:", result.moving_average.size)
flat = analyze_time_series(TimeSeriesInput([2, 2, 2], window=1))
print("плоский ряд:", flat.local_maxima_indices, flat.local_minima_indices, flat.moving_average)

print("\n=== Задача 8. one_hot ===")
print(one_hot(OneHotInput(labels=[0, 2, 3, 0])))                 # class_count по max метке
print(one_hot(OneHotInput(labels=[0, 2, 3, 0], class_count=5)))  # ровно 5 столбцов
print(one_hot(OneHotInput(labels=[0, 0, 0], class_count=1)))
print(one_hot(OneHotInput(labels=[])))                           # пустой вход
