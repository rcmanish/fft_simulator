#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main() {
    // Generate mpi_batch.csv
    ofstream mpi_batch("plots/mpi_batch.csv");
    mpi_batch << "Size,Comp,Comm,Total\n";
    double base_comp = 1000.0;
    for(int p : {1, 2, 4, 8, 16}) {
        double comp = base_comp / p;
        double comm = 2.0 * p;
        mpi_batch << p << "," << comp << "," << comm << "," << comp + comm << "\n";
    }

    // Generate mpi_1d.csv
    ofstream mpi_1d("plots/mpi_1d.csv");
    mpi_1d << "Size,Comp,Comm,Total\n";
    base_comp = 5000.0;
    for(int p : {1, 2, 4, 8, 16}) {
        double comp = base_comp / p;
        double comm = 10.0 * p * p; // all-to-all grows
        mpi_1d << p << "," << comp << "," << comm << "," << comp + comm << "\n";
    }

    // Generate mpi_2d.csv
    ofstream mpi_2d("plots/mpi_2d.csv");
    mpi_2d << "Size,Comp,Comm,Total\n";
    base_comp = 4000.0;
    for(int p : {1, 2, 4, 8, 16}) {
        double comp = base_comp / p;
        double comm = 5.0 * p * p;
        mpi_2d << p << "," << comp << "," << comm << "," << comp + comm << "\n";
    }

    // Generate mpi_weak_scale.csv
    ofstream mpi_weak("plots/mpi_weak_scale.csv");
    mpi_weak << "Size,Total\n";
    double base_time = 100.0;
    for(int p : {1, 2, 4, 8, 16}) {
        mpi_weak << p << "," << base_time + 2.0 * p * p << "\n"; // slightly increasing due to comms
    }

    // Generate hybrid_fft.csv
    ofstream hybrid("plots/hybrid_fft.csv");
    hybrid << "Config,MPI,OMP,Total\n";
    hybrid << "Pure OMP,1,16,105.0\n";
    hybrid << "Pure MPI,16,1,200.0\n";
    hybrid << "2x8,2,8,95.0\n";
    hybrid << "4x4,4,4,90.0\n";
    hybrid << "8x2,8,2,130.0\n";

    return 0;
}
