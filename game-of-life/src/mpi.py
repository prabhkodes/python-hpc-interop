import numpy as np
from mpi4py import MPI
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ON = 1
OFF = 0
STEPS = 50
N = 100

def count_neighbors_padded(grid):
    val_rows = grid.shape[0] - 2
    N  = grid[0:-2, :]
    S  = grid[2:  , :]
    W  = np.roll(grid[1:-1, :],  1, axis=1)
    E  = np.roll(grid[1:-1, :], -1, axis=1)
    NW = np.roll(grid[0:-2, :],  1, axis=1)
    NE = np.roll(grid[0:-2, :], -1, axis=1)
    SW = np.roll(grid[2:  , :],  1, axis=1)
    SE = np.roll(grid[2:  , :], -1, axis=1)
    return N + S + E + W + NW + NE + SW + SE

def step_game(grid):
    neighbors = count_neighbors_padded(grid)
    center_cells = grid[1:-1, :]
    new_center = np.zeros_like(center_cells)
    births = (center_cells == OFF) & (neighbors == 3)
    survivals = (center_cells == ON) & ((neighbors == 2) | (neighbors == 3))
    new_center[births | survivals] = ON
    return new_center

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rows_per_rank = N // size
    local_data = np.zeros((rows_per_rank, N), dtype=int)
    full_grid = None
    history = []

    if rank == 0:
        np.random.seed(42)
        full_grid = np.random.choice([ON, OFF], N*N, p=[0.2, 0.8]).reshape(N, N)
        history.append(full_grid.copy())
    
    comm.Scatter(full_grid, local_data, root=0)
    local_grid_padded = np.zeros((rows_per_rank + 2, N), dtype=int)
    local_grid_padded[1:-1, :] = local_data
    up_neighbor = (rank - 1) % size
    down_neighbor = (rank + 1) % size

    for _ in range(STEPS):
        send_up = np.ascontiguousarray(local_grid_padded[1, :])
        send_down = np.ascontiguousarray(local_grid_padded[-2, :])
        recv_top = np.empty(N, dtype=int)
        recv_bottom = np.empty(N, dtype=int)

        comm.Sendrecv(sendbuf=send_up, dest=up_neighbor, recvbuf=recv_bottom, source=down_neighbor)
        comm.Sendrecv(sendbuf=send_down, dest=down_neighbor, recvbuf=recv_top, source=up_neighbor)

        local_grid_padded[0, :] = recv_top
        local_grid_padded[-1, :] = recv_bottom
        local_grid_padded[1:-1, :] = step_game(local_grid_padded)

        snap_local = local_grid_padded[1:-1, :].copy()
        snap_full = None
        if rank == 0:
            snap_full = np.zeros((N, N), dtype=int)
        comm.Gather(snap_local, snap_full, root=0)
        if rank == 0:
            history.append(snap_full)

    if rank == 0:
        fig, ax = plt.subplots()
        img = ax.imshow(history[0], cmap='binary')
        def update(frame):
            img.set_data(history[frame])
            return [img]
        ani = animation.FuncAnimation(fig, update, frames=len(history), interval=50)
        ani.save('game_of_life.gif', writer='pillow')
        plt.close()

if __name__ == "__main__":
    main()