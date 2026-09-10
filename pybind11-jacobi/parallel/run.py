import jacobi_mpi as jm
import sys

def main():
    # 1. Configuration (Replaces your .in file parsing)
    N = 10
    CORNER = 100.0
    STEPS = 100
    
    # 2. Initialize Mesh
    mesh = jm.CMesh(N, CORNER, STEPS)
    rank = mesh.world_rank
    size = mesh.world_size

    if rank == 0:
        print(f"Mesh Initialized: {N}x{N} global interior.")

    # # 3. Print Initial Local Grids (Ordered by Rank)
    # for r in range(size):
    #     if rank == r:
    #         print(f"\n===== Rank {rank} (num_rows={mesh.N1}, N={mesh.N2}) =====")
    #         print("[old_field]")
    #         jm.print_grid_local(mesh.old_field, mesh.N1, mesh.N2, 1, 6)
    #     jm.barrier()

    # 4. Solve
    mesh.jacobi_solver()

    # 5. Collect Statistics
    jm.print_performance_stats()

    # # 6. Print Final Local Grids (Ordered by Rank)
    # jm.barrier()
    # if rank == 0:
    #     print("\n" + "="*40 + "\nFINAL RESULTS\n" + "="*40)
        
    # for r in range(size):
    #     if rank == r:
    #         print(f"\n===== Rank {rank} Final State =====")
    #         jm.print_grid_local(mesh.old_field, mesh.N1, mesh.N2, 1, 6)
    #     jm.barrier()

if __name__ == "__main__":
    main()