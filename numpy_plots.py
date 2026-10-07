import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grader_contracts.numpy_tasks import MatrixStatistics, RandomMatrixInput
from numpy_tasks import matrix_statistics

def _plot_vector_histograms(matrix: np.ndarray, title_prefix: str, figure_title: str,
                            means: np.ndarray, variances: np.ndarray, color: str,
                            bins: int, path: str | None) -> None:
    """Сетка гистограмм: по одной на каждый одномерный срез матрицы."""
    count = matrix.shape[0]
    columns = min(4, count)
    rows = int(np.ceil(count / columns))

    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
    for index in range(count):
        axis = axes[index // columns][index % columns]
        axis.hist(matrix[index], bins=bins, color=color, edgecolor="black", linewidth=0.4)
        axis.set_title(f"{title_prefix} {index}: mean={means[index]:.3f}, "
                       f"var={variances[index]:.3f}", fontsize=9)
        axis.set_xlabel("value", fontsize=8)
        axis.set_ylabel("count", fontsize=8)
        axis.tick_params(labelsize=7)
    for index in range(count, rows * columns):  # прячем пустые ячейки сетки
        axes[index // columns][index % columns].axis("off")

    figure.suptitle(figure_title)
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_row_histograms(stats: MatrixStatistics, bins: int = 25, path: str | None = None) -> None:
    """Гистограмма значений для каждой строки матрицы."""
    matrix = np.asarray(stats.matrix)
    _plot_vector_histograms(matrix, "Row", "Histograms of matrix rows",
                            np.asarray(stats.row_means), np.asarray(stats.row_variances),
                            "steelblue", bins, path)


def plot_column_histograms(stats: MatrixStatistics, bins: int = 25, path: str | None = None) -> None:
    """Гистограмма значений для каждого столбца матрицы (срезы столбцов = строки транспонированной)."""
    matrix = np.asarray(stats.matrix).T
    _plot_vector_histograms(matrix, "Column", "Histograms of matrix columns",
                            np.asarray(stats.column_means), np.asarray(stats.column_variances),
                            "darkorange", bins, path)


if __name__ == "__main__":
    # Для наглядных гистограмм строк берём матрицу с длинными строками,
    # для гистограмм столбцов — с длинными столбцами.
    wide = matrix_statistics(RandomMatrixInput(rows=6, columns=500, mean=5.0, std=1.5, seed=42))
    plot_row_histograms(wide, path="row_histograms.png")

    tall = matrix_statistics(RandomMatrixInput(rows=500, columns=6, mean=5.0, std=1.5, seed=42))
    plot_column_histograms(tall, path="column_histograms.png")

    print("Сохранено: row_histograms.png, column_histograms.png")
    print("row_means (wide):      ", np.round(wide.row_means, 4))
    print("column_means (wide):   ", np.round(wide.column_means[:6], 4), "...")
    print("row_variances (wide):  ", np.round(wide.row_variances, 4))
    print("column_variances (tall):", np.round(tall.column_variances, 4))
