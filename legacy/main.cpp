#include <vector>
#include <iostream>

#include "rambo.h"

int main() {

    int iterations = 2;

    double et = 100.0;
    std::vector<double> masses = {10.0, 20.0};
    double weight;

    for (int i = 0; i < iterations; ++i) {

        std::vector<double*> momenta = rambo(et, masses, weight);
        
        std::cout << "Generated Momenta:" << std::endl;
        for (size_t i = 0; i < momenta.size(); ++i) {
            std::cout << "p" << i << ": (";
            for (size_t j = 0; j < 4; ++j) {
                std::cout << momenta[i][j];
                if (j < 3) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
            std::cout << "p" << i << "^2: " << momenta[i][0]*momenta[i][0] - momenta[i][1]*momenta[i][1] 
                      - momenta[i][2]*momenta[i][2] - momenta[i][3]*momenta[i][3] << std::endl;
        }
        double *pTot = new double[4]{0.0, 0.0, 0.0, 0.0}; 
        for (size_t i = 0; i < momenta.size(); ++i) {
            pTot[0] += momenta[i][0];
            pTot[1] += momenta[i][1];
            pTot[2] += momenta[i][2];
            pTot[3] += momenta[i][3];
        }
        std::cout << "Total Momentum^2: " << pTot[0]*pTot[0] - pTot[1]*pTot[1] - pTot[2]*pTot[2] - pTot[3]*pTot[3] << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            std::cout << "Total p[" << i << "]: " << pTot[i] << std::endl;
        }
        delete[] pTot;

        std::cout << "Weight: " << weight << std::endl;
        std::cout << "------------------------" << std::endl;
    }
    
    return 0;
}