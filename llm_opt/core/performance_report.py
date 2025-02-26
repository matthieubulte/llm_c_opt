from typing import List, Dict, Any
import numpy as np


class PerformanceReport:
    def __init__(self, quantiles: List[float] = [0.025, 0.25, 0.5, 0.75, 0.975]):
        self.c_times = []
        self.numpy_times = []
        self.quantiles = quantiles

    def add_c_runtime(self, runtime: float):
        self.c_times.append(runtime)

    def add_numpy_runtime(self, runtime: float):
        self.numpy_times.append(runtime)

    def calculate_c_quantiles(self) -> Dict[float, float]:
        return dict(zip(self.quantiles, np.percentile(self.c_times, self.quantiles)))

    def calculate_numpy_quantiles(self) -> Dict[float, float]:
        return dict(
            zip(self.quantiles, np.percentile(self.numpy_times, self.quantiles))
        )

    def avg_speedup(self) -> float:
        return np.mean(self.numpy_times) / np.mean(self.c_times)

    def avg_c_speed(self) -> float:
        return np.mean(self.c_times)

    def avg_numpy_speed(self) -> float:
        return np.mean(self.numpy_times)

    def desc(self) -> str:
        c_quantiles = self.calculate_c_quantiles()
        numpy_quantiles = self.calculate_numpy_quantiles()

        c_quantiles_formatted = {
            f"{q*100:.1f}%": f"{v*1000:.4f} ms" for q, v in c_quantiles.items()
        }
        numpy_quantiles_formatted = {
            f"{q*100:.1f}%": f"{v*1000:.4f} ms" for q, v in numpy_quantiles.items()
        }

        return f"""
    Performance Summary:
    -------------------
    Speedup: {self.avg_speedup():.2f}x (E[NumPy]/E[C])

    Runtime Statistics:
    C Implementation:
        • Average: {self.avg_c_speed()*1000:.4f} ms
        • Quantiles: {', '.join([f"{k}: {v}" for k, v in c_quantiles_formatted.items()])}
        
    NumPy Implementation:
        • Average: {self.avg_numpy_speed()*1000:.4f} ms
        • Quantiles: {', '.join([f"{k}: {v}" for k, v in numpy_quantiles_formatted.items()])}
    """
