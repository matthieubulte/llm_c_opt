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

gcc main.c -o main -O3 -march=native -ftree-vectorize -ffast-math -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas -framework Accelerate

================================================================================ FINAL INSTRUCTIONS
Provide an optimized C implementation that is numerically equivalent to the NumPy function.
Do not worry about performance.
Do not forget to import any necessary libraries.
Return ONLY the C function implementation in ```c ... ``` tags, not the entire file.
"""


def gen_feedback_prompt(numpy_source, artifacts_str):
    return f"""
An LLM loop is currently trying to optimize the C implementation of the following python function

```python
{numpy_source}
```

The target LLM does not have access to any tool or can control the compiling process. It can only provide the C implementation.

SYSTEM INFO
{SYSTEM_INFO}

COMPILATION COMMAND
gcc main.c -o main -O3 -march=native -ftree-vectorize -ffast-math -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas -framework Accelerate

TASK
I want you to take the feedback from the previous iterations and provide list of takeaways and suggestions for the next iteration.
Your goal is to help the LLM loop to find the best possible C implementation, not to implement the function yourself.
The structure of the feedback should be as follows:

```
## Bugs that were encountered
...

## Approaches to Avoid
...

## Next steps
...
```

Only answer with the feedback in the format above.

START OF RUN HISTORY
```
{artifacts_str}
```
END OF RUN HISTORY
"""


def gen_update_prompt(
    numpy_source, function_signature, artifacts_str, feedback_str=None
):
    if feedback_str is None:
        feedback_str = """
Consider the following optimization techniques:

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
"""

    return f"""
You are the world's leading expert on optimizing C code, with years of experience in both numerical linear algebra and C programming. Your task is to translate the following NumPy function to an optimized C implementation.

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

gcc main.c -o main -O3 -march=native -ftree-vectorize -ffast-math -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas -framework Accelerate

================================================================================ FEEDBACK
{feedback_str}

================================================================================ FINAL INSTRUCTIONS
Provide an improved implementation.
Do not forget to import any necessary libraries.
Return ONLY the C function implementation in ```c ... ``` tags, not the entire file.
Do not repeat the same implementation as in the previous iterations. Any repeated implementation will be penalized.
If you realize that the implementation is not improving, be creative and explore new ways to optimize the code.
Make sure to pay attention to the previous iterations and their errors and performance analysis. Very important!!!
"""
