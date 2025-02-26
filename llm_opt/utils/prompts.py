def gen_initial_prompt(numpy_source, function_signature):
    return f"""
        I need you to translate the following NumPy function to an optimized C implementation:
        
        ```python
{numpy_source}
        ```
        
        The C function should have the following signature:
        ```c
{function_signature}
        ```
        
        Please provide an optimized C implementation that is numerically equivalent to the NumPy function. Do not worry about performance.
        
        Return ONLY the C function implementation, not the entire file.
        """


def gen_update_prompt(numpy_source, function_signature, artifacts_str):
    return f"""
            I need you to translate the NumPy function to an optimized C implementation.
            
            NumPy function:
            ```python
{numpy_source}
            ```
            
            C function signature:
            ```c
{function_signature}
            ```
            
            Here are artifacts from previous iterations:
{artifacts_str}
            
            Consider exploring these optimization techniques:
            
            Vectorization:
            - Use SIMD instructions (SSE, AVX, NEON) for parallel operations
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

            
            Please provide an improved implementation.
            Return ONLY the C function implementation in ```c ... ``` tags, not the entire file.
            """
