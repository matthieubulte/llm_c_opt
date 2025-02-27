from dotenv import load_dotenv

load_dotenv()

import numpy as np
from llm_opt import DeepSeekAPIClient, GroqAPIClient
from llm_opt.core.signature import Signature
from llm_opt.core.type_interface import DOUBLE, INT
from llm_opt.utils.helpers import ensure_directory_exists
from llm_opt.core.optimizer import Optimizer


def foo(a, b, out):
    out[0] = np.dot(a**2, np.sqrt(b)) / np.sum(a)


def foo_inputs():
    n = 100_000
    a = np.random.rand(n).astype(np.float64)
    b = np.random.rand(n).astype(np.float64)
    out = np.zeros(1).astype(np.float64)
    return (a, b, out)


def monte_carlo_pi(x, y, out):
    acc = 0
    nsamples = x.shape[0]
    for i in range(nsamples):
        if (x[i] ** 2 + y[i] ** 2) < 1.0:
            acc += 1
    out[0] = 4.0 * acc / nsamples


def monte_carlo_pi_inputs():
    nsamples = 100_000
    x = np.random.rand(nsamples).astype(np.float64)
    y = np.random.rand(nsamples).astype(np.float64)
    out = np.zeros(1).astype(np.float64)
    return (x, y, out)


def matrix_convolution(input_matrix: np.ndarray, kernel: np.ndarray, out: np.ndarray):
    """
    Performs 2D convolution of input_matrix with kernel.
    Both arguments are 2D numpy arrays.
    """
    n = int(np.sqrt(input_matrix.shape[0]))
    k = int(np.sqrt(kernel.shape[0]))

    input_matrix = input_matrix.reshape(n, n)
    kernel = kernel.reshape(k, k)
    out = out.reshape(n - k + 1, n - k + 1)

    for i in range(n - k + 1):
        for j in range(n - k + 1):
            window = input_matrix[i : i + k, j : j + k]
            out[i, j] = np.sum(window * kernel)


def matrix_convolution_inputs():
    n = 100
    k = 3
    input_matrix = np.random.rand(n * n).astype(np.float64)
    kernel = np.random.rand(k * k).astype(np.float64)
    out = np.zeros((n - k + 1) * (n - k + 1)).astype(np.float64)
    return (input_matrix, kernel, out)


def main():
    output_dir = "results"
    ensure_directory_exists(output_dir)
    # api_client = GroqAPIClient()
    api_client = DeepSeekAPIClient()

    test_case = "matrix_convolution"
    get_feedback = False
    max_iterations = 20
    benchmark_runs = 250

    if test_case == "matrix_convolution":
        Optimizer(
            matrix_convolution,
            Signature(
                [
                    ("x", DOUBLE.array_of()),
                    ("y", DOUBLE.array_of()),
                    ("out", DOUBLE.array_of()),
                ]
            ),
            matrix_convolution_inputs,
            api_client=api_client,
            max_iterations=max_iterations,
            benchmark_runs=benchmark_runs,
            get_feedback=get_feedback,
        ).optimize()

    elif test_case == "foo":
        Optimizer(
            foo,
            Signature(
                [
                    ("a", DOUBLE.array_of()),
                    ("b", DOUBLE.array_of()),
                    ("out", DOUBLE.array_of()),
                ]
            ),
            foo_inputs,
            api_client=api_client,
            err_tol=1e-6,
            max_iterations=max_iterations,
            benchmark_runs=benchmark_runs,
            output_dir=output_dir,
            get_feedback=get_feedback,
        ).optimize()


if __name__ == "__main__":
    main()
