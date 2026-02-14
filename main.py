import random
import math
import numpy as np
from functools import lru_cache

def generate_random_data(size=1000000):
    """Generate large random dataset using NumPy for efficiency"""
    return np.random.random(size)

def perform_calculations(data):
    """Perform heavy mathematical operations using vectorization"""
    return np.sum(np.sin(data) * np.cos(data) + np.sqrt(np.abs(data)))

def matrix_operations(n=1000):
    """Create and manipulate large matrices"""
    matrix = np.random.rand(n, n)
    return np.linalg.matrix_rank(matrix)

@lru_cache(maxsize=None)
def recursive_function(n=100):
    """Recursive calculations with memoization"""
    if n <= 0:
        return 1
    return n * recursive_function(n - 1)

if __name__ == "__main__":
    print("Generating random data...")
    data = generate_random_data(1000000)
    
    print("Performing calculations...")
    result = perform_calculations(data)
    print(f"Calculation result: {result:.2f}")
    
    print("Matrix operations...")
    rank = matrix_operations(500)
    print(f"Matrix rank: {rank}")
    
    print("Recursive calculation...")
    fac = recursive_function(50)
    print(f"Result: {fac}")
