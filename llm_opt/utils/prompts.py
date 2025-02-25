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


def gen_update_prompt(numpy_source, function_signature, c_implementation, feedback):
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
            
            Your previous implementation was:
            ```c
{c_implementation}
            ```
            
            Here's feedback on your implementation:
            
            {feedback}
            
            Please provide an improved implementation.
            Return ONLY the C function implementation, not the entire file.
            """
