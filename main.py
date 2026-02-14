import random
import math
import numpy as np

def generate_random_data(size=1000000):
    """Generate large random dataset"""
    return [random.random() for _ in range(size)]

def perform_calculations(data):
    """Perform heavy mathematical operations"""
    result = 0
    for value in data:
        result += math.sin(value) * math.cos(value) + math.sqrt(abs(value))
    return result

def matrix_operations(n=1000):
    """Create and manipulate large matrices"""
    matrix = np.random.rand(n, n)
    return np.linalg.matrix_rank(matrix)

def recursive_function(n=100):
    """Recursive calculations"""
    if n <= 0:
        return 1
    return n * recursive_function(n - 1)

if __name__ == "__main__":
    print("Generating random data...")
    data = generate_random_data(1000000)
    
    print("Performing calculations...")
    result = perform_calculations(data)
    print(f"Calculation result: {result}")
    
    print("Matrix operations...")
    rank = matrix_operations(500)
    print(f"Matrix rank: {rank}")
    
    print("Recursive calculation...")
    fac = recursive_function(50)
    print(f"Result: {fac}")