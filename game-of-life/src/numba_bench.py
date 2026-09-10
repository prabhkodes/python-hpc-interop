import numpy as np
import time
from numba import jit, njit, stencil, prange

@jit
def evolve_jit(grid, N):
    """
    Exercise 1a: Manual loop implementation decorated with @jit.
    """
    temp_grid = grid.copy()
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            # Count neighbors
            total = (grid[i-1, j-1] + grid[i-1, j] + grid[i-1, j+1] +
                     grid[i, j-1]                + grid[i, j+1] +
                     grid[i+1, j-1] + grid[i+1, j] + grid[i+1, j+1])
            
            # Apply Rules
            if grid[i, j] == 1:
                if total < 2 or total > 3:
                    temp_grid[i, j] = 0
            else:
                if total == 3:
                    temp_grid[i, j] = 1
    
    # Copy back results
    for i in range(N):
        for j in range(N):
            grid[i, j] = temp_grid[i, j]

@njit(parallel=True) 
def evolve_njit(grid, N):
    """
    Parallel version of your NJIT implementation.
    """
    temp_grid = grid.copy()
    
    
    for i in prange(1, N-1): 
        for j in range(1, N-1):
            total = (grid[i-1, j-1] + grid[i-1, j] + grid[i-1, j+1] +
                     grid[i, j-1]                + grid[i, j+1] +
                     grid[i+1, j-1] + grid[i+1, j] + grid[i+1, j+1])
            
            if grid[i, j] == 1:
                if total < 2 or total > 3:
                    temp_grid[i, j] = 0
            else:
                if total == 3:
                    temp_grid[i, j] = 1
    
    
    for i in prange(N):
        for j in range(N):
            grid[i, j] = temp_grid[i, j]

@jit(forceobj=True)
def evolve_slicing_jit(grid):
    """
    Exercise 1b: Applying JIT to the Numpy slicing method.
    
    """
    # Axis 0 = Rows (Up/Down)
    N_val = np.roll(grid, -1, axis=0)
    S     = np.roll(grid,  1, axis=0)
    
    # Axis 1 = Columns (Left/Right)
    E     = np.roll(grid, -1, axis=1)
    W     = np.roll(grid,  1, axis=1)
    
    # Diagonals
    NE    = np.roll(N_val, -1, axis=1)
    NW    = np.roll(N_val,  1, axis=1)
    SE    = np.roll(S,     -1, axis=1)
    SW    = np.roll(S,      1, axis=1)
    
    neighbors = N_val + S + E + W + NE + NW + SE + SW
    
    # Apply Rules
    grid[:] = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)
    grid[:] = np.where((grid == 0) & (neighbors == 3), 1, grid)

@stencil
def evolve_stencil(grid):
    """
    Exercise 1c: Numba Stencil implementation.
    """
    # Sum of 8 neighbors relative to center (0,0)
    neighbors = (grid[-1, -1] + grid[-1, 0] + grid[-1, 1] +
                 grid[ 0, -1]               + grid[ 0, 1] +
                 grid[ 1, -1] + grid[ 1, 0] + grid[ 1, 1])
    
    if grid[0, 0] == 1:
        if neighbors < 2 or neighbors > 3:
            return 0 
        else:
            return 1 
    else:
        if neighbors == 3:
            return 1 
        else:
            return 0 


class GameOfLifeBenchmark:
    def __init__(self, size=400, steps=50):
        print(f"Initializing Game of Life Benchmark (Grid: {size}x{size}, Steps: {steps})...\n")
        self.size = size
        self.steps = steps
        np.random.seed(42)
        self.grid = np.random.randint(2, size=(size, size), dtype=np.int32)

    def evolve_numpy(self, grid):
        """Standard Numpy implementation (Reference)"""
        N  = np.roll(grid, -1, axis=0)
        S  = np.roll(grid,  1, axis=0)
        E  = np.roll(grid, -1, axis=1)
        W  = np.roll(grid,  1, axis=1)
        NE = np.roll(N, -1, axis=1)
        NW = np.roll(N,  1, axis=1)
        SE = np.roll(S, -1, axis=1)
        SW = np.roll(S,  1, axis=1)
        
        neighbors = N + S + E + W + NE + NW + SE + SW
        
        grid[:] = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)
        grid[:] = np.where((grid == 0) & (neighbors == 3), 1, grid)

    def run_benchmark(self):
        """Run all requested variations."""
        
        # 1. Baseline
        print("--- 1. Baseline (Numpy Slicing) ---")
        grid_copy = self.grid.copy()
        start = time.time()
        for _ in range(self.steps):
            self.evolve_numpy(grid_copy)
        end = time.time()
        print(f"Time: {end - start:.4f} seconds\n")

        # 2. Exercise 1a: JIT vs NJIT on Loops
        print("--- 2. Exercise 1a (JIT vs NJIT on Loops) ---")
        
        # JIT
        grid_copy = self.grid.copy()
        evolve_jit(grid_copy, self.size) # Warmup
        start = time.time()
        for _ in range(self.steps):
            evolve_jit(grid_copy, self.size)
        end = time.time()
        print(f"JIT Loop Time:  {end - start:.4f} seconds")

        # NJIT
        grid_copy = self.grid.copy()
        evolve_njit(grid_copy, self.size) # Warmup
        start = time.time()
        for _ in range(self.steps):
            evolve_njit(grid_copy, self.size)
        end = time.time()
        print(f"NJIT Loop Time: {end - start:.4f} seconds")
        

        # 3. Exercise 1b: JIT on Slicing
        print("--- 3. Exercise 1b (JIT on Slicing) ---")
        grid_copy = self.grid.copy()
        # Note: We use forceobj=True so this may not be faster than Numpy
        evolve_slicing_jit(grid_copy) # Warmup
        start = time.time()
        for _ in range(self.steps):
            evolve_slicing_jit(grid_copy)
        end = time.time()
        print(f"JIT Slicing Time: {end - start:.4f} seconds")
        

        # 4. Exercise 1c: Stencil
        print("--- 4. Exercise 1c (Numba Stencil) ---")
        grid_copy = self.grid.copy()
        _ = evolve_stencil(grid_copy) # Warmup
        
        start = time.time()
        for _ in range(self.steps):
            # Stencil returns a new array, unlike the others which are in-place
            grid_copy = evolve_stencil(grid_copy)
        end = time.time()
        print(f"Stencil Time:     {end - start:.4f} seconds")

if __name__ == "__main__":
    app = GameOfLifeBenchmark(size=1000, steps=100)
    app.run_benchmark()