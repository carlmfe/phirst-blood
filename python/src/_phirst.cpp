#ifndef PHIRST_BACKEND_SERIAL
#define PHIRST_BACKEND_SERIAL
#endif

#include <phirst/phirst.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace phirst;

namespace {

template <typename Integrand, int N>
IntegrationResult run_for_n(const Integrand& integ,
                            double cmEnergy,
                            const nb::ndarray<const double, nb::numpy, nb::shape<-1>, nb::c_contig>& massesArr,
                            int64_t nEvents,
                            uint64_t seed,
                            bool useVegas) {
    if (massesArr.ndim() != 1 || static_cast<int>(massesArr.shape(0)) != N) {
        throw std::invalid_argument("masses must be a 1D float64 array with length matching n_particles");
    }

    double masses[N] = {};
    auto massesView = massesArr.view();
    for (int i = 0; i < N; ++i) {
        masses[i] = massesView(i);
    }

    RamboIntegrator<Integrand, N> integrator(nEvents, integ);
    IntegrationResult result;
    result.nEvents = nEvents;

    if (useVegas) {
        integrator.runVegas(cmEnergy, masses, result.mean, result.error, seed);
    } else {
        integrator.run(cmEnergy, masses, result.mean, result.error, seed);
    }

    return result;
}

template <typename Integrand>
IntegrationResult dispatch_integrate(const Integrand& integ,
                                     int nParticles,
                                     double cmEnergy,
                                     const nb::ndarray<const double, nb::numpy, nb::shape<-1>, nb::c_contig>& masses,
                                     int64_t nEvents,
                                     uint64_t seed,
                                     bool useVegas) {
    switch (nParticles) {
        case 2:
            return run_for_n<Integrand, 2>(integ, cmEnergy, masses, nEvents, seed, useVegas);
        case 3:
            return run_for_n<Integrand, 3>(integ, cmEnergy, masses, nEvents, seed, useVegas);
        case 4:
            return run_for_n<Integrand, 4>(integ, cmEnergy, masses, nEvents, seed, useVegas);
        case 5:
            return run_for_n<Integrand, 5>(integ, cmEnergy, masses, nEvents, seed, useVegas);
        case 6:
            return run_for_n<Integrand, 6>(integ, cmEnergy, masses, nEvents, seed, useVegas);
        default:
            throw std::invalid_argument("n_particles must be 2-6");
    }
}

template <int N>
void bind_mandelstam_dispatcher(nb::module_& m, const char* name) {
    m.def(
        name,
        [](const MandelstamSIntegrand<N>& integ,
           double cmEnergy,
           const nb::ndarray<const double, nb::numpy, nb::shape<-1>, nb::c_contig>& masses,
           int64_t nEvents,
           uint64_t seed,
           bool useVegas) {
            nb::gil_scoped_release release;
            return run_for_n<MandelstamSIntegrand<N>, N>(integ, cmEnergy, masses, nEvents, seed, useVegas);
        },
        nb::arg("integrand"),
        nb::arg("cm_energy"),
        nb::arg("masses"),
        nb::arg("n_events"),
        nb::arg("seed"),
        nb::arg("use_vegas"));
}

}  // namespace

NB_MODULE(_phirst, m) {
    m.doc() = "Phirst Blood: Monte Carlo phase space integration";

    nb::class_<IntegrationResult>(m, "IntegrationResult")
        .def_ro("mean", &IntegrationResult::mean)
        .def_ro("error", &IntegrationResult::error)
        .def_ro("n_events", &IntegrationResult::nEvents)
        .def("__repr__",
             [](const IntegrationResult& r) {
                 return "IntegrationResult(mean=" + std::to_string(r.mean) +
                        ", error=" + std::to_string(r.error) +
                        ", n_events=" + std::to_string(r.nEvents) + ")";
             });

    nb::class_<ConstantIntegrand>(m, "ConstantIntegrand")
        .def(nb::init<double>(), nb::arg("value") = 1.0);

    nb::class_<DrellYanIntegrand>(m, "DrellYanIntegrand")
        .def(nb::init<double, double>(),
             nb::arg("charge") = 2.0 / 3.0,
             nb::arg("coupling") = 1.0 / 137.035999);

    nb::class_<EggholderIntegrand>(m, "EggholderIntegrand")
        .def(nb::init<double>(), nb::arg("lambda_squared") = 1000000.0);

    nb::class_<MandelstamSIntegrand<2>>(m, "MandelstamSIntegrand2")
        .def(nb::init<double>(), nb::arg("scale") = 1.0);
    nb::class_<MandelstamSIntegrand<3>>(m, "MandelstamSIntegrand3")
        .def(nb::init<double>(), nb::arg("scale") = 1.0);
    nb::class_<MandelstamSIntegrand<4>>(m, "MandelstamSIntegrand4")
        .def(nb::init<double>(), nb::arg("scale") = 1.0);
    nb::class_<MandelstamSIntegrand<5>>(m, "MandelstamSIntegrand5")
        .def(nb::init<double>(), nb::arg("scale") = 1.0);
    nb::class_<MandelstamSIntegrand<6>>(m, "MandelstamSIntegrand6")
        .def(nb::init<double>(), nb::arg("scale") = 1.0);

    m.def(
        "_integrate_constant",
        [](const ConstantIntegrand& integ,
           int nParticles,
           double cmEnergy,
           const nb::ndarray<const double, nb::numpy, nb::shape<-1>, nb::c_contig>& masses,
           int64_t nEvents,
           uint64_t seed,
           bool useVegas) {
            nb::gil_scoped_release release;
            return dispatch_integrate(integ, nParticles, cmEnergy, masses, nEvents, seed, useVegas);
        },
        nb::arg("integrand"),
        nb::arg("n_particles"),
        nb::arg("cm_energy"),
        nb::arg("masses"),
        nb::arg("n_events"),
        nb::arg("seed"),
        nb::arg("use_vegas"));

    m.def(
        "_integrate_drell_yan",
        [](const DrellYanIntegrand& integ,
           int nParticles,
           double cmEnergy,
           const nb::ndarray<const double, nb::numpy, nb::shape<-1>, nb::c_contig>& masses,
           int64_t nEvents,
           uint64_t seed,
           bool useVegas) {
            nb::gil_scoped_release release;
            return dispatch_integrate(integ, nParticles, cmEnergy, masses, nEvents, seed, useVegas);
        },
        nb::arg("integrand"),
        nb::arg("n_particles"),
        nb::arg("cm_energy"),
        nb::arg("masses"),
        nb::arg("n_events"),
        nb::arg("seed"),
        nb::arg("use_vegas"));

    m.def(
        "_integrate_eggholder",
        [](const EggholderIntegrand& integ,
           int nParticles,
           double cmEnergy,
           const nb::ndarray<const double, nb::numpy, nb::shape<-1>, nb::c_contig>& masses,
           int64_t nEvents,
           uint64_t seed,
           bool useVegas) {
            nb::gil_scoped_release release;
            return dispatch_integrate(integ, nParticles, cmEnergy, masses, nEvents, seed, useVegas);
        },
        nb::arg("integrand"),
        nb::arg("n_particles"),
        nb::arg("cm_energy"),
        nb::arg("masses"),
        nb::arg("n_events"),
        nb::arg("seed"),
        nb::arg("use_vegas"));

    bind_mandelstam_dispatcher<2>(m, "_integrate_mandelstam2");
    bind_mandelstam_dispatcher<3>(m, "_integrate_mandelstam3");
    bind_mandelstam_dispatcher<4>(m, "_integrate_mandelstam4");
    bind_mandelstam_dispatcher<5>(m, "_integrate_mandelstam5");
    bind_mandelstam_dispatcher<6>(m, "_integrate_mandelstam6");
}
