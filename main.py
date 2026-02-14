import numpy as np
from functools import lru_cache
import time
from collections import defaultdict, deque
import heapq

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

# DSA: Graph algorithms
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
    
    def add_edge(self, u, v, weight=1):
        self.graph[u].append((v, weight))
        self.vertices.add(u)
        self.vertices.add(v)
    
    def dijkstra(self, start):
        """Dijkstra's shortest path algorithm"""
        distances = {v: float('inf') for v in self.vertices}
        distances[start] = 0
        pq = [(0, start)]
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            if current_dist > distances[u]:
                continue
            for v, weight in self.graph[u]:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    heapq.heappush(pq, (distances[v], v))
        return distances
    
    def bfs(self, start):
        """Breadth-first search"""
        visited = set()
        queue = deque([start])
        visited.add(start)
        result = []
        
        while queue:
            u = queue.popleft()
            result.append(u)
            for v, _ in self.graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return result

# DSA: Binary Search Tree
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if val < node.val:
            if node.left is None:
                node.left = TreeNode(val)
            else:
                self._insert_recursive(node.left, val)
        else:
            if node.right is None:
                node.right = TreeNode(val)
            else:
                self._insert_recursive(node.right, val)
    
    def inorder_traversal(self):
        result = []
        self._inorder(self.root, result)
        return result
    
    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.val)
            self._inorder(node.right, result)

# DSA: Sorting algorithms
def merge_sort(arr):
    """Merge sort algorithm"""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr):
    """Quick sort algorithm"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# DSA: Dynamic Programming
@lru_cache(maxsize=None)
def longest_increasing_subsequence_length(arr, n, prev=-1):
    """LIS using DP"""
    if n == 0:
        return 0
    incl = 0
    if prev == -1 or arr[n-1] > arr[prev]:
        incl = 1 + longest_increasing_subsequence_length(arr, n-1, n-1)
    excl = longest_increasing_subsequence_length(arr, n-1, prev)
    return max(incl, excl)

@lru_cache(maxsize=None)
def edit_distance(s1, s2, m, n):
    """Edit distance (Levenshtein) using DP"""
    if m == 0:
        return n
    if n == 0:
        return m
    if s1[m-1] == s2[n-1]:
        return edit_distance(s1, s2, m-1, n-1)
    return 1 + min(
        edit_distance(s1, s2, m-1, n),
        edit_distance(s1, s2, m, n-1),
        edit_distance(s1, s2, m-1, n-1)
    )

# DSA: Hash Map operations
def frequency_counter(arr):
    """Count frequency using hash map"""
    freq = defaultdict(int)
    for num in arr:
        freq[num] += 1
    return freq

if __name__ == "__main__":
    start_time = time.time()
    
    print("=== NumPy Operations ===")
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
    
    print("Eigenvalue decomposition...")
    eig_sum = eigenvalue_decomposition(1500)
    print(f"Eigenvalue sum: {eig_sum:.2f}")
    
    print("\n=== DSA: Graph ===")
    g = Graph()
    for i in range(100):
        g.add_edge(i, (i+1) % 100, np.random.randint(1, 10))
    distances = g.dijkstra(0)
    print(f"Dijkstra distances from 0: {len(distances)} vertices")
    
    print("\n=== DSA: BST ===")
    bst = BST()
    for _ in range(1000):
        bst.insert(np.random.randint(1, 10000))
    print(f"BST inorder (first 10): {bst.inorder_traversal()[:10]}")
    
    print("\n=== DSA: Sorting ===")
    test_arr = list(np.random.randint(0, 1000, 5000))
    sorted_merge = merge_sort(test_arr[:100])
    sorted_quick = quick_sort(test_arr[:100])
    print(f"Merge sort result (first 10): {sorted_merge[:10]}")
    
    print("\n=== DSA: Dynamic Programming ===")
    fib = fibonacci(35)
    print(f"Fibonacci(35): {fib}")
    
    print("\n=== DSA: Hash Map ===")
    freq = frequency_counter(test_arr[:1000])
    print(f"Unique elements: {len(freq)}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f}s")

# tis code demonstrates a comprehensive set of operations including NumPy for data manipulation, graph algorithms, binary search tree operations, sorting algorithms, dynamic programming techniques, and hash map usage. It is designed to test the performance and efficiency of various algorithms and data structures while handling large datasets.