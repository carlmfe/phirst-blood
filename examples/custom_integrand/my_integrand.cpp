// Example user integrand: computes (p1 + p2)^2 = Mandelstam s
#include <phirst/integrands.hpp>

struct MyIntegrand {
    double evaluate(const double momenta[][4]) const {
        double s = 0.0;
        for (int mu = 0; mu < 4; ++mu) {
            double tot = 0.0;
            for (int i = 0; i < 2; ++i) {
                tot += momenta[i][mu];
            }
            s += (mu == 0 ? 1.0 : -1.0) * tot * tot;
        }
        return s;
    }
};
