from llm_opt.utils.constants import SYSTEM_INFO


def gen_initial_prompt(numpy_source, function_signature):
    return f"""
Your task is to translate the following NumPy function to a correct C implementation.


================================================================================ NUMPY FUNCTION
```python
{numpy_source}
```

================================================================================ C FUNCTION SIGNATURE
```c
{function_signature}
```

================================================================================ SYSTEM INFO
{SYSTEM_INFO}

================================================================================ COMPILATION COMMAND

gcc main.c -o main -O3 -march=native -ftree-vectorize -ffast-math -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas

================================================================================ FINAL INSTRUCTIONS
Provide an optimized C implementation that is numerically equivalent to the NumPy function.
Do not worry about performance.
Do not forget to import any necessary libraries.
Return ONLY the C function implementation in ```c ... ``` tags, not the entire file.
"""


def gen_update_prompt(numpy_source, function_signature, artifacts_str):
    return f"""
Your task is to translate the following NumPy function to an optimized C implementation.

================================================================================ NUMPY FUNCTION
```python
{numpy_source}
```
            
================================================================================ C FUNCTION SIGNATURE
```c
{function_signature}
```

================================================================================ PREVIOUS ITERATIONS

{artifacts_str}            

================================================================================ SYSTEM INFO

{SYSTEM_INFO}

================================================================================ COMPILATION COMMAND

gcc main.c -o main -O3 -march=native -ftree-vectorize -ffast-math -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas

================================================================================ OPTIMIZATION TECHNIQUES

Vectorization:
- Use SIMD instructions (you're running on an M1 chip) for parallel operations
- Align data to vector boundaries for efficient SIMD access
- Vectorize inner loops where possible

Memory access optimizations:
- Implement cache blocking/tiling to improve cache utilization
- Optimize data layout for better spatial locality
- Use prefetching to reduce cache misses

Loop optimizations:
- Unroll loops to reduce branch prediction misses
- Fuse loops to reduce loop overhead
- Interchange loops to improve memory access patterns

Algorithm improvements:
- Look for mathematical simplifications
- Consider specialized algorithms for common patterns
- Reduce redundant computation
- Use the best possible algorithm for the problem

Each of these categories is important and should be considered.

================================================================================ FINAL INSTRUCTIONS
Provide an improved implementation.
Do not forget to import any necessary libraries.
Return ONLY the C function implementation in ```c ... ``` tags, not the entire file.
Do not repeat the same implementation as in the previous iterations. Any repeated implementation will be penalized.
If you realize that the implementation is not improving, be creative and explore new ways to optimize the code.
"""
