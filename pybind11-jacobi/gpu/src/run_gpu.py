import cupy as cp
import numpy as np
from mpi4py import MPI
import time

class GPUMesh:
    def __init__(self, N, corner_value, max_steps):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Ensure each rank uses a unique GPU
        cp.cuda.Device(self.rank % 4).use() 
        
        base = N // self.size
        rem = N % self.size
        self.N1 = base + (1 if self.rank < rem else 0)
        self.N2 = N
        
        self.rows, self.cols = self.N1 + 2, self.N2 + 2
        
        # GPU Arrays
        self.old_field = cp.zeros((self.rows, self.cols), dtype=cp.float64)
        self.new_field = cp.zeros((self.rows, self.cols), dtype=cp.float64)
        
        # CPU Staging buffers for stable MPI communication
        self.send_buffer = np.zeros(self.cols - 2, dtype=np.float64)
        self.recv_buffer = np.zeros(self.cols - 2, dtype=np.float64)
        
        self.corner_value = corner_value
        self.max_steps = max_steps
        self.apply_boundary()

    def apply_boundary(self):
        if self.rank == 0:
            self.old_field[0, :] = 0.0
        if self.rank == self.size - 1:
            self.old_field[-1, :] = cp.linspace(self.corner_value, 0, self.cols)
        self.old_field[:, -1] = 0.0
        
        start_row_idx = self.comm.scan(self.N1) - self.N1
        for i in range(1, self.N1 + 1):
            global_row = start_row_idx + i
            self.old_field[i, 0] = self.corner_value * (global_row / self.N2)
        
        self.new_field[:] = self.old_field[:]

    def exchange_boundaries(self):
        up = self.rank - 1 if self.rank > 0 else MPI.PROC_NULL
        down = self.rank + 1 if self.rank < self.size - 1 else MPI.PROC_NULL

        # Exchange UP
        self.send_buffer[:] = cp.asnumpy(self.old_field[1, 1:-1])
        self.comm.Sendrecv(self.send_buffer, dest=up, recvbuf=self.recv_buffer, source=up)
        self.old_field[0, 1:-1] = cp.array(self.recv_buffer)

        # Exchange DOWN
        self.send_buffer[:] = cp.asnumpy(self.old_field[-2, 1:-1])
        self.comm.Sendrecv(self.send_buffer, dest=down, recvbuf=self.recv_buffer, source=down)
        self.old_field[-1, 1:-1] = cp.array(self.recv_buffer)

    def jacobi_solver(self):
        if self.rank == 0: print(f"Starting Solver...")
        self.comm.Barrier()
        t_start = time.perf_counter()
        
        for step in range(self.max_steps):
            self.exchange_boundaries()
            # GPU Slicing
            self.new_field[1:-1, 1:-1] = 0.25 * (
                self.old_field[0:-2, 1:-1] + self.old_field[2:, 1:-1] +
                self.old_field[1:-1, 0:-2] + self.old_field[1:-1, 2:]
            )
            self.old_field, self.new_field = self.new_field, self.old_field
            
        cp.cuda.Stream.null.synchronize()
        if self.rank == 0: print(f"Done in {time.perf_counter()-t_start:.4f}s")

if __name__ == "__main__":
    mesh = GPUMesh(N=1000, corner_value=100.0, max_steps=500)
    mesh.jacobi_solver()