// This would be my fourth commit. I have changed the code from using the Finite Difference(FD) Algorithm to Fast Fourier Transform(FFT) algorithm, which 
// much faster than FD. This would help in increasing the integration time to much higher values.

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip> 
#include <unsupported/Eigen/MatrixFunctions>
# include <unsupported/Eigen/FFT>
using namespace std;

//-------------------------------------FUNCTION DECLARATIONS--------------------------------------------

// Harmonic Trap declaration
double trap_potential(double x_val);

//Split operator step for spinless BEC
void split_operator_step(Eigen::VectorXcd& Psi,
        Eigen::VectorXd& trap_potential_array,Eigen::VectorXcd& Opr_K,
        double dt, double c0, Eigen::FFT<double>& fft);

// -------------------------------------FUNCTION DEFINITIONS--------------------------------------------
// Harmonic trap definition
double trap_potential(double x_val){
    double pot = 0.5 * pow(x_val,2.0);
    return pot; 
}

// // Split operator step 
void split_operator_step(Eigen::VectorXcd& Psi,
        Eigen::VectorXd& trap_potential_array,Eigen::VectorXcd& Opr_K,
        double dt, double c0, Eigen::FFT<double>& fft){ 

    Eigen::VectorXd V_eff = trap_potential_array + c0*Psi.cwiseAbs2();
    complex<double> I_half_dt(0.0,-dt/2.0);
    Eigen::VectorXcd Opr_V = (I_half_dt*V_eff.cast<complex<double>>()).array().exp();
    Psi = Opr_V.cwiseProduct(Psi);
    //Fourier transform Psi here to Phi through some FFTW?? 
    Eigen::VectorXcd Phi(Psi.size());
    fft.fwd(Phi,Psi);
    // Then probably have to define Opr_K: exp(-idt*p**2/2m)
    Phi = Opr_K.cwiseProduct(Phi); 
    // Inverse fourier transform to get Psi
    fft.inv(Psi,Phi);
    Psi = Opr_V.cwiseProduct(Psi);
}

int main(){
    // Parameters
    int Nx = 256;
    double x_min = -10;
    double x_max = 10;
    double L = x_max - x_min;
    double dx = (x_max - x_min)/double(Nx);
    double pi = acos(-1.0);
    double c0 = 0.01;
    double dt = 0.001;
    double T = 10*pi;
    double Nt = static_cast<int>(std::round(T/dt));
    int center_idx = Nx / 2;

    // Grid Position Space
    Eigen::VectorXd x(Nx);
    for (int i = 0; i < Nx ; i++){
        x(i) = x_min + i*dx;}
    
    // Grid Momentum Space
    Eigen::VectorXd k(Nx);
    double dk = (2.0 * pi)/L;
    for (int i= 0; i<Nx;i++){
        if (i < Nx/2){
            k[i] = i *dk; // Positive Frequencies
        } else {
            k[i] = (i - Nx)*dk; // Negative frequencies
        }
    }

    // Potential
    Eigen::VectorXd trap_potential_array(Nx);
    for (int j = 0; j<Nx;j++){
        trap_potential_array[j] = trap_potential(x[j]);
    }

    // Exponential of the Kinetic Operator in the split step method
    complex<double> I_dt(0.0,dt);
    Eigen::VectorXd T_array = 0.5 * k.array().square();
    Eigen::VectorXcd Opr_K = (-1.0 * I_dt * T_array.cast<complex<double>>()).array().exp();

    // Initial Wavefunction
    Eigen::VectorXcd Psi(Nx);
    double analytic_norm = pow(1.0/pi,0.25);
    for (int i = 0; i < Nx; i++){
        double val = analytic_norm * exp(-0.5*pow(x[i],2.0));
        Psi[i] = complex<double>(val,0.0);
    }
    // Numerical Normalisation check
    double norm_sq = Psi.cwiseAbs2().sum() * dx;
    Psi /= sqrt(norm_sq);

    // FFT setup
    Eigen::FFT<double> fft;

    // --- CSV File Setup ---
    ofstream density_file("center_density_using_FFT.csv");
    if (!density_file.is_open()) {
        cerr << "Error opening file!" << endl;
        return 1;
    }
    // CSV Header
    density_file << "Time,Center_Density" << endl;
    cout<<"Sarting the Split Step simulation using FFT"<<endl;
    
    for (int step = 0;step < Nt ; step++){
        double current_time = step * dt;
        // Save central density (magnitude squared)
        double central_density = std::norm(Psi(center_idx)); 
        density_file << fixed << setprecision(6) << current_time << "," 
                     << scientific << setprecision(8) << central_density << "\n";
        split_operator_step(Psi, trap_potential_array, Opr_K, dt,c0,fft);

        if (step % 1000 == 0) {
            // Norm check in real space (Parseval's theorem guarantees it in k-space too)
            double current_norm = Psi.cwiseAbs2().sum() * dx;
            cout << "Step: " << step << " | Norm: " << current_norm << endl;
        }
    }
   density_file.close();
    cout << "Simulation Complete. Data saved to center_density.csv" << endl;

    // Save final density profile to CSV
    ofstream final_psi("final_density_using_FFT.csv");
    final_psi << "x,Density" << endl;
    for (int i = 0; i < Nx; ++i) {
        final_psi << x(i) << "," << std::norm(Psi(i)) << endl;
    }
    final_psi.close();

    return 0;
}