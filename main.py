import numpy as np
from functools import lru_cache
import time

def generate_random_data(size=50000000):
    """Generate large random dataset using NumPy for efficiency"""
    return np.random.random(size)

def generate_random_matrix(rows=5000, cols=5000):
    """Generate random matrix with specified dimensions"""
    return np.random.rand(rows, cols)

def perform_calculations(data):
    """Perform heavy mathematical operations using vectorization"""
    return np.sum(np.sin(data) * np.cos(data) + np.sqrt(np.abs(data)))

def advanced_calculations(data):
    """Additional mathematical operations"""
    return np.mean(np.log(data + 1)), np.std(data), np.max(data)

def matrix_operations(n=5000):
    """Create and manipulate large matrices"""
    matrix = np.random.rand(n, n)
    return np.linalg.matrix_rank(matrix)

def matrix_multiplication(n=2000):
    """Perform matrix multiplication"""
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    return np.matmul(a, b)

def eigenvalue_decomposition(n=2000):
    """Calculate eigenvalues of a matrix"""
    matrix = np.random.rand(n, n)
    eigenvalues, _ = np.linalg.eig(matrix)
    return np.sum(eigenvalues)

@lru_cache(maxsize=None)
def recursive_function(n=100):
    """Recursive calculations with memoization"""
    if n <= 0:
        return 1
    return n * recursive_function(n - 1)

@lru_cache(maxsize=None)
def fibonacci(n=35):
    """Calculate fibonacci with memoization"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

if __name__ == "__main__":
    start_time = time.time()
    
    print("Generating random data...")
    data = generate_random_data(50000000)
    
    print("Performing calculations...")
    result = perform_calculations(data)
    print(f"Calculation result: {result:.2f}")
    
    print("Advanced calculations...")
    mean, std, max_val = advanced_calculations(data)
    print(f"Mean: {mean:.2f}, Std: {std:.2f}, Max: {max_val:.2f}")
    
    print("Matrix operations...")
    rank = matrix_operations(500)
    print(f"Matrix rank: {rank}")
    
    print("Matrix multiplication...")
    product = matrix_multiplication(1500)
    print(f"Multiplication result shape: {product.shape}")
    
    print("Eigenvalue decomposition...")
    eig_sum = eigenvalue_decomposition(1500)
    print(f"Eigenvalue sum: {eig_sum:.2f}")
    
    print("Recursive calculation...")
    fac = recursive_function(50)
    print(f"Factorial result: {fac}")
    
    print("Fibonacci calculation...")
    fib = fibonacci(35)
    print(f"Fibonacci result: {fib}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f}s")
