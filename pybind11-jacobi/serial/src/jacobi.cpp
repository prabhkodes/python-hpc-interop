#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

namespace py = pybind11;

// ------------------------------------------------------------------
// 1. Opaque Vector Binding
// ------------------------------------------------------------------
PYBIND11_MAKE_OPAQUE(std::vector<double>);

// ------------------------------------------------------------------
// 2. Boundary Condition Function (Converted from Template to double)
// ------------------------------------------------------------------
void default_boundary_condition(std::vector<double>& vec, int N, double corner, double field_value) {
    int dim = N + 2; 
    int last = dim - 1; 

    // Fill field value
    std::fill(vec.begin(), vec.end(), field_value); 

    auto idx = [dim](int i, int j){ return i * dim + j; };

    double step = corner / static_cast<double>(last); 

    // Fill top row with zeros
    for (int j = 0; j < dim; ++j)
        vec[idx(0, j)] = 0.0;

    // Fill Last column with zeros
    for (int i = 0; i < dim; ++i)
        vec[idx(i, last)] = 0.0;

    // Fill bottom row, linearly dec
    for (int j = 0; j < dim; ++j)
        vec[idx(last, j)] = corner - step * static_cast<double>(j);

    // Fill first Column, linearly dec
    for (int i = 0; i < dim; ++i)
        vec[idx(i, 0)] = step * static_cast<double>(i);
}

// ------------------------------------------------------------------
// 3. CMesh Class
// ------------------------------------------------------------------
class CMesh {
public:
    int N;
    std::vector<double> field;
    std::vector<double> new_field;

    CMesh(int n,
          std::function<void(std::vector<double>&, int, double, double)> boundary_conditions,
          double corner,
          double field_value) 
    {
        N = n;
        int dim = N + 2;
        field.resize(dim * dim);
        new_field.resize(dim * dim);

        // Apply BC
        boundary_conditions(field, N, corner, field_value);

        // Init new_field
        new_field = field;
    }

    friend std::ostream& operator<<(std::ostream& os, const CMesh& m) {
        int dim = m.N + 2;
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                os << std::setfill(' ') << std::setw(8) 
                   << std::fixed << std::setprecision(2)
                   << m.field[i * dim + j] << " ";
            }
            os << std::endl;
        }
        return os;
    }
};

// ------------------------------------------------------------------
// 4. CSolver Class (Converted from Template to double)
// ------------------------------------------------------------------
class CSolver {
public:
    void jacobi(CMesh& m, const int& max_steps, const int& PrintInterval) {
        int dim  = m.N + 2;
        int last = dim - 1;

        for (int step = 1; step <= max_steps; ++step) {
            
            // Jacobi Iteration
            for (int i = 1; i < last; ++i) {
                for (int j = 1; j < last; ++j) {
                    m.new_field[i*dim + j] =
                        (m.field[(i-1)*dim + j] +
                         m.field[i*dim + (j-1)] +
                         m.field[i*dim + (j+1)] +
                         m.field[(i+1)*dim + j]) * 0.25;
                }
            }

            m.field.swap(m.new_field);

            // Print Logic (Replaced external printer with std::cout)
            if (PrintInterval > 0 && (step % PrintInterval == 0)) {
                std::cout << "Step " << step << ":\n" << m << "\n" << std::endl;
            }
        }
    }
};

// ------------------------------------------------------------------
// 5. Pybind11 Module Definition
// ------------------------------------------------------------------
PYBIND11_MODULE(jacobi, m) {
    // Bind Vector
    py::bind_vector<std::vector<double>>(m, "VectorDouble");

    // Expose the C++ default boundary function
    m.def("default_boundary_condition", &default_boundary_condition, 
          "Default linear gradient boundary condition");

    // Bind CMesh
    py::class_<CMesh>(m, "CMesh")
        .def(py::init<int, std::function<void(std::vector<double>&, int, double, double)>, double, double>(),
             py::arg("n"), 
             py::arg("boundary_conditions"), 
             py::arg("corner"), 
             py::arg("field_value"))
        .def_readwrite("N", &CMesh::N)
        .def_readwrite("field", &CMesh::field)
        .def("__str__", [](const CMesh &mesh) {
            std::stringstream ss;
            ss << mesh;
            return ss.str();
        });

    // Bind CSolver
    py::class_<CSolver>(m, "CSolver")
        .def(py::init<>())
        .def("jacobi", &CSolver::jacobi, 
             py::arg("mesh"), 
             py::arg("max_steps"), 
             py::arg("print_interval") = 0);
}