#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "mesh.hpp"
#include "printer.hpp"
#include "timer.hpp"

namespace py = pybind11;

// 1. Make the vector opaque for performance
PYBIND11_MAKE_OPAQUE(std::vector<double>);

// ------------------------------------------------------------------
// Pybind11 Module Definition
// ------------------------------------------------------------------
PYBIND11_MODULE(jacobi_mpi, m) {
    m.doc() = "Parallel Jacobi Solver: MPI + OpenMP + Pybind11";

    // --- MPI Lifecycle Management ---
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }

    // --- Bind VectorDouble ---
    py::bind_vector<std::vector<double>>(m, "VectorDouble");

    // --- Bind CMesh <double> ---
    using MeshD = CMesh<double>;
    py::class_<MeshD>(m, "CMesh")
        .def(py::init<long int, double, long>(),
             py::arg("N"), 
             py::arg("corner"), 
             py::arg("steps"))
        
        .def_readonly("world_rank", &MeshD::world_rank)
        .def_readonly("world_size", &MeshD::world_size)
        .def_readonly("N1", &MeshD::N1)
        .def_readonly("N2", &MeshD::N2)
        .def_readonly("rows", &MeshD::rows)
        .def_readonly("cols", &MeshD::cols)
        
        .def("jacobi_solver", &MeshD::jacobi_solver)
        .def_readwrite("old_field", &MeshD::old_field)
        .def_readwrite("new_field", &MeshD::new_field);

    // --- Bind Printer Utilities (from printer.hpp) ---
    // We wrap these so they can be called easily from Python
    m.def("print_grid_local", [](const std::vector<double>& field, long num_rows, long N, int precision, int width) {
        utils::print_grid_local<double>(field, num_rows, N, precision, width);
    }, "Prints the local MPI rank's portion of the grid");

    // --- Bind Timer Utilities (from timer.hpp) ---
    m.def("print_performance_stats", []() {
        std::vector<TimerData> all_timings;
        CTimer::gather_and_print(0, all_timings);
    }, "Gathers MPI timings from all ranks and writes to statistics.txt");

    // --- Helper: MPI Barrier ---
    m.def("barrier", []() {
        MPI_Barrier(MPI_COMM_WORLD);
    }, "Synchronize all MPI processes");

    // --- Cleanup ---
    auto cleanup_callback = []() {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    };
    m.add_object("_cleanup", py::capsule(cleanup_callback));
}