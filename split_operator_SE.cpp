// This would be my second commit. This is a basic code using split operator and finite difference method to solve a non-linear schrodinger equation.

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip> 
#include <unsupported/Eigen/MatrixFunctions>

// Exponential of Kinetic Operator using Matrix Exponential
Eigen::MatrixXcd Kinetic_Operator(const Eigen::MatrixXcd& K_matrix, double dt) {
    std::complex<double> I_dt(0.0, -dt);
    Eigen::MatrixXcd M = K_matrix * I_dt;
    return M.exp();
}

double V_trap(double x_val) {
    return 0.5 * (std::pow(x_val, 2));
}

int main() {
    // --- Constants and Parameters ---
    const double x_min = -10;
    const double x_max = 10;
    const int Nx = 256;
    const double PI = acos(-1.0);
    double dx = (x_max - x_min) / double(Nx);
    const double dt = 0.001;
    double g_int = 0.01;
    double TOTAL_TIME = 10.0 * PI;
    const int N_STEPS = static_cast<int>(std::round(TOTAL_TIME / dt));

    // --- Grid Initialization ---
    Eigen::VectorXd x(Nx);
    for (int j = 0; j < Nx; j++) {
        x[j] = x_min + j * dx;
    }

    // --- Potential and Kinetic Matrix ---
    Eigen::VectorXd V_trap_array(Nx);
    Eigen::MatrixXcd K_matrix(Nx, Nx);
    K_matrix.setZero();
    double coeff = -0.5 / (dx * dx);

    for (int i = 0; i < Nx; i++) {
        V_trap_array[i] = V_trap(x[i]);
        K_matrix(i, i) = std::complex<double>(-2.0 * coeff, 0.0);
        if (i < Nx - 1) K_matrix(i, i + 1) = std::complex<double>(1.0 * coeff, 0.0);
        if (i > 0)      K_matrix(i, i - 1) = std::complex<double>(1.0 * coeff, 0.0);
    }

    // Pre-calculate the full kinetic propagator matrix
    Eigen::MatrixXcd kinetic_propagator = Kinetic_Operator(K_matrix, dt);

    // --- Initial Wavefunction (Ground State) ---
    Eigen::VectorXcd Psi(Nx);
    double norm_sum = 0;
    for (int i = 0; i < Nx; i++) {
        double psi_real = std::pow(1.0 / PI, 0.25) * std::exp(-0.5 * std::pow(x[i], 2));
        Psi[i] = std::complex<double>(psi_real, 0.0);
        norm_sum += std::norm(Psi[i]) * dx;
    }
    Psi /= std::sqrt(norm_sum); // Normalize

    std::ofstream file_time("Finite_difference_center_density_evolution.csv");
    file_time << "time,center_density" << std::endl;

    std::cout << "Starting Finite Difference Split-Step simulation..." << std::endl;

    // Time Loop
    for (int step = 0; step < N_STEPS; step++) {
        double current_time = step * dt;

        //central density at current time
        double density_center = std::norm(Psi[Nx / 2]);
        file_time << std::fixed << std::setprecision(6) << current_time << "," 
                  << std::scientific << std::setprecision(8) << density_center << "\n";

        //Split-Step Algorithm
        // Half Potential Step
        Eigen::VectorXd V_eff = V_trap_array + g_int * Psi.cwiseAbs2();
        std::complex<double> I_half_dt(0.0, -dt / 2.0);
        Eigen::VectorXcd P_V = (I_half_dt * V_eff.cast<std::complex<double>>()).array().exp();
        
        Psi = P_V.cwiseProduct(Psi);

        // Full Kinetic Step 
        Psi = kinetic_propagator * Psi;

        // Final Half Potential Step
        V_eff = V_trap_array + g_int * Psi.cwiseAbs2(); // Re-calculate V_eff with new Psi
        P_V = (I_half_dt * V_eff.cast<std::complex<double>>()).array().exp();
        Psi = P_V.cwiseProduct(Psi);

        if (step % 5000 == 0) {
            std::cout << "Time: " << current_time << " | Center Density: " << density_center << std::endl;
        }
    }
    file_time.close();

    // --- Save Final Density Profile ---
    std::ofstream file_final("Finite_difference_final_density_profile.csv");
    file_final << "x,density" << std::endl;
    for (int i = 0; i < Nx; i++) {
        file_final << std::fixed << std::setprecision(6) << x[i] << "," 
                   << std::scientific << std::setprecision(8) << std::norm(Psi[i]) << "\n";
    }
    file_final.close();

    return 0;
}