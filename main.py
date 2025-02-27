import numpy as np
from llm_opt import DeepSeekAPIClient
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


def main():
    output_dir = "results"
    ensure_directory_exists(output_dir)
    Optimizer(
        monte_carlo_pi,
        Signature(
            [
                ("x", DOUBLE.array_of()),
                ("y", DOUBLE.array_of()),
                ("out", DOUBLE.array_of()),
            ]
        ),
        monte_carlo_pi_inputs,
        api_client=DeepSeekAPIClient(),
        max_iterations=20,
    ).optimize()
    return
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
        api_client=DeepSeekAPIClient(),
        err_tol=1e-6,
        max_iterations=10,
        benchmark_runs=250,
        output_dir=output_dir,
    ).optimize()
    return
    Optimizer(
        bar,
        Signature(
            [
                ("n", INT),
                ("k", INT),
                ("x", DOUBLE.array_of()),
                ("weights", DOUBLE.array_of()),
                ("scaling_factor", DOUBLE),
                ("out", DOUBLE.array_of()),
            ]
        ),
        bar_inputs,
        api_client=DeepSeekAPIClient(),
        err_tol=1e-6,
        max_iterations=20,
        benchmark_runs=200,
        output_dir=output_dir,
    ).optimize()


if __name__ == "__main__":
    main()
