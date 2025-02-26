import numpy as np
from llm_opt import DeepSeekAPIClient
from llm_opt.core.signature import Signature
from llm_opt.core.type_interface import DOUBLE, INT
from llm_opt.utils.helpers import ensure_directory_exists
from llm_opt.core.optimizer import Optimizer


def foo(a, b, out):
    out[0] = np.dot(a**2, np.sqrt(b)) / np.sum(a)


def foo_inputs():
    n = 1000
    a = np.random.rand(n).astype(np.float64)
    b = np.random.rand(n).astype(np.float64)
    out = np.zeros(1).astype(np.float64)
    return (a, b, out)


def bar(a, n, k, out):
    a = a.reshape((n, n))
    tmp = a
    for _ in range(k):
        tmp = a @ tmp
    out[:] = tmp.mean()


def bar_inputs():
    n = 10
    k = np.random.randint(1, 5)
    a = np.random.rand(n * n).reshape((n, n)).astype(np.float64)
    out = np.zeros(1).astype(np.float64)
    return (a, n, k, out)


def main():
    output_dir = "results"
    ensure_directory_exists(output_dir)
    # Optimizer(
    #     foo,
    #     Signature(
    #         [
    #             ("a", DOUBLE.array_of()),
    #             ("b", DOUBLE.array_of()),
    #             ("out", DOUBLE.array_of()),
    #         ]
    #     ),
    #     foo_inputs,
    #     api_client=DeepSeekAPIClient(),
    #     err_tol=1e-6,
    #     max_iterations=10,
    #     benchmark_runs=100,
    #     output_dir=output_dir,
    # ).optimize()

    Optimizer(
        bar,
        Signature(
            [
                ("a", DOUBLE.array_of()),
                ("n", INT),
                ("k", INT),
                ("out", DOUBLE.array_of()),
            ]
        ),
        bar_inputs,
        api_client=DeepSeekAPIClient(),
        err_tol=1e-6,
        max_iterations=10,
        benchmark_runs=100,
        output_dir=output_dir,
    ).optimize()


if __name__ == "__main__":
    main()
