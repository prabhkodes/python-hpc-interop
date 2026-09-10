import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ON = 1
OFF = 0
Vals = [ON, OFF]

def count_neighbors(grid):
    N  = np.roll(grid, -1, axis=0)
    S  = np.roll(grid,  1, axis=0)
    E  = np.roll(grid, -1, axis=1)
    W  = np.roll(grid,  1, axis=1)
    NE = np.roll(grid, (-1, -1), axis=(0, 1))
    NW = np.roll(grid, (-1,  1), axis=(0, 1))
    SE = np.roll(grid, ( 1, -1), axis=(0, 1))
    SW = np.roll(grid, ( 1,  1), axis=(0, 1))

    return N + S + E + W + NE + NW + SE + SW

def update(frameNum, img, grid, N):
    neighbors = count_neighbors(grid)
    new_grid = grid.copy()

    births = (grid == OFF) & (neighbors == 3)
    survivals = (grid == ON) & ((neighbors == 2) | (neighbors == 3))
    
    new_grid[:] = OFF
    new_grid[births | survivals] = ON

    grid[:] = new_grid[:]
    img.set_data(new_grid)
    return img,

def main():
    N = 100 
    grid = np.random.choice(Vals, N*N, p=[0.2, 0.8]).reshape(N, N)

    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap='binary')
    plt.title("Conway's Game of Life")
    
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N),
                                  frames=50, interval=50)

    ani.save('game_of_life.gif', writer='pillow', fps=20)
    plt.close(fig)

if __name__ == '__main__':
    main()