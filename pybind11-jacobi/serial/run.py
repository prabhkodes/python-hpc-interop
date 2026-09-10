import jacobi
import time

# 1. Define the parameters
N = 50                 # Grid size (excluding boundaries)
CORNER_TEMP = 100.0    # Temperature at corners
INITIAL_TEMP = 10.0    # Starting temperature inside
STEPS = 1000           # Number of Jacobi iterations

print(f"Creating mesh of size {N}x{N}...")

# 2. Initialize the Mesh
# We pass the C++ default boundary function (exposed via pybind11)
mesh = jacobi.CMesh(
    n=N, 
    boundary_conditions=jacobi.default_boundary_condition, 
    corner=CORNER_TEMP, 
    field_value=INITIAL_TEMP
)

# 3. Initialize the Solver
solver = jacobi.CSolver()

# 4. Run the simulation
print("Starting simulation...")
start_time = time.time()

# Run solver (prints grid every 0 steps, meaning never, to save screen space)
solver.jacobi(mesh, max_steps=STEPS, print_interval=0)

end_time = time.time()
print(f"Done in {end_time - start_time:.4f} seconds.")

# 5. Visualize the result (Text)
# We can print small meshes directly, but for large ones, let's peek at the corner
# The mesh object behaves like the C++ class
print("\nTop-left corner of the grid (first 5x5):")
# Access the flat 'field' vector directly from Python
dim = N + 2
for i in range(5):
    row_vals = mesh.field[i*dim : i*dim + 5]
    print([f"{val:.1f}" for val in row_vals])