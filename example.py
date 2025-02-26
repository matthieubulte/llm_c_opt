import numpy as np
from llm_opt import DeepSeekAPIClient
from llm_opt.core.signature import Signature
from llm_opt.core.type_interface import DOUBLE
from llm_opt.utils.helpers import ensure_directory_exists
from llm_opt.core.optimizer import Optimizer


def vec_add(a, b, out):
    out[0] = np.linalg.norm(a + b)


def vec_add_inputs():
    n = 10_000_000
    a = np.random.rand(n).astype(np.float64)
    b = np.random.rand(n).astype(np.float64)
    out = np.zeros(1).astype(np.float64)
    return (a, b, out)


def main():
    output_dir = "results"
    ensure_directory_exists(output_dir)
    Optimizer(
        vec_add,
        Signature(
            [
                ("a", DOUBLE.array_of()),
                ("b", DOUBLE.array_of()),
                ("out", DOUBLE.array_of()),
            ]
        ),
        vec_add_inputs,
        DeepSeekAPIClient(),
        5,
        100,
        output_dir,
    ).optimize()


if __name__ == "__main__":
    main()
