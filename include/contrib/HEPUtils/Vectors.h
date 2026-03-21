// -*- C++ -*-
//
// This file is part of HEPUtils -- https://gitlab.com/hepcedar/heputils/
// Copyright (C) 2013-2023 Andy Buckley <andy.buckley@cern.ch>
//
// Embedding of HEPUtils code in other projects is permitted provided this
// notice is retained and the HEPUtils namespace and include path are changed.
//
#pragma once

#include "MathUtils.h"
#include "Utils.h"
#if defined(PHIRST_BACKEND_SERIAL)
#include <sstream>
#include <iostream>
#include <stdexcept>
#endif

/// @file Physics vectors stuff
/// @author Andy Buckley <andy.buckley@cern.ch>

namespace HEPUtils {


  /// @brief A robust 4-momentum class for on-shell vectors.
  ///
  /// P4 is a typical 4-momentum class, cf. HepLorentzVector or TLorentzVector
  /// with the exception that the data member storage is specifically based on
  /// (px, py, pz, m) rather than (px, py, pz, E). This means that there are (or
  /// at least should) never be numerical precision problems due to calculations
  /// like m^2 = E^2 - p^2 when p^2 >> m^2. In this form, m^2 is always
  /// well-defined and the "equivalent" calculation is the numerically safe E^2
  /// = p^2 + m^2.
  ///
  /// This design restricts usage to on-shell vectors... which as far as I'm
  /// aware is not really a restriction in practice, at last not for physical
  /// particles, but please let me know if it is! (Off-shell vectors require a
  /// 5th component so the spatial and time components can be inconsistent with
  /// the mass.)
  ///
  class P4 {
  private:

    /// @name Storage
    //@{
    double _x, _y, _z, _m;
    //@}


  public:

    /// @name Constructors etc.
    //@{

    /// Default constructor of a null vector
    PHIRST_HOST_DEVICE
    P4()
      : _x(0), _y(0), _z(0), _m(0) {  }

    /// Constructor from Cartesian/Minkowski coordinates
    PHIRST_HOST_DEVICE
    P4(double px, double py, double pz, double E) {
      setPE(px, py, pz, E);
    }

    /// Copy constructor
    PHIRST_HOST_DEVICE
    P4(const P4& v)
      : _x(v._x), _y(v._y), _z(v._z), _m(v._m) {  }

    /// Copy assignment operator
    PHIRST_HOST_DEVICE
    P4& operator = (const P4& v) {
      _x = v._x;
      _y = v._y;
      _z = v._z;
      _m = v._m;
      return *this;
    }

    /// Set the components to zero
    PHIRST_HOST_DEVICE
    void clear() {
      _x = 0;
      _y = 0;
      _z = 0;
      _m = 0;
    }

    //@}


    /// @name Static methods for vector making
    //@{

    /// Make a vector from (px,py,pz,E) coordinates
    PHIRST_HOST_DEVICE
    static P4 mkXYZE(double px, double py, double pz, double E) {
      return P4().setPE(px, py, pz, E);
    }

    /// Make a vector from (px,py,pz) coordinates and the mass
    PHIRST_HOST_DEVICE
    static P4 mkXYZM(double px, double py, double pz, double mass) {
      return P4().setPM(px, py, pz, mass);
    }

    /// Make a vector from (eta,phi,energy) coordinates and the mass
    PHIRST_HOST_DEVICE
    static P4 mkEtaPhiME(double eta, double phi, double mass, double E) {
      return P4().setEtaPhiME(eta, phi, mass, E);
    }

    /// Make a vector from (eta,phi,pT) coordinates and the mass
    PHIRST_HOST_DEVICE
    static P4 mkEtaPhiMPt(double eta, double phi, double mass, double pt) {
      return P4().setEtaPhiMPt(eta, phi, mass, pt);
    }

    /// Make a vector from (y,phi,energy) coordinates and the mass
    PHIRST_HOST_DEVICE
    static P4 mkRapPhiME(double y, double phi, double mass, double E) {
      return P4().setRapPhiME(y, phi, mass, E);
    }

    /// Make a vector from (y,phi,pT) coordinates and the mass
    PHIRST_HOST_DEVICE
    static P4 mkRapPhiMPt(double y, double phi, double mass, double pt) {
      return P4().setRapPhiMPt(y, phi, mass, pt);
    }

    /// Make a vector from (theta,phi,energy) coordinates and the mass
    PHIRST_HOST_DEVICE
    static P4 mkThetaPhiME(double theta, double phi, double mass, double E) {
      return P4().setThetaPhiME(theta, phi, mass, E);
    }

    /// Make a vector from (theta,phi,pT) coordinates and the mass
    PHIRST_HOST_DEVICE
    static P4 mkThetaPhiMPt(double theta, double phi, double mass, double pt) {
      return P4().setThetaPhiMPt(theta, phi, mass, pt);
    }

    /// Make a vector from (pT,phi,energy) coordinates and the mass
    PHIRST_HOST_DEVICE
    static P4 mkPtPhiME(double pt, double phi, double mass, double E) {
      return P4().setPtPhiME(pt, phi, mass, E);
    }

    //@}


    /// @name Coordinate setters
    //@{

    /// Set the px coordinate
    PHIRST_HOST_DEVICE
    P4& setPx(double px) {
      _x = px;
      return *this;
    }

    /// Set the py coordinate
    PHIRST_HOST_DEVICE
    P4& setPy(double py) {
      _y = py;
      return *this;
    }

    /// Set the pz coordinate
    PHIRST_HOST_DEVICE
    P4& setPz(double pz) {
      _z = pz;
      return *this;
    }

    /// Set the mass
    PHIRST_HOST_DEVICE
    P4& setM(double mass) {
#if defined(PHIRST_BACKEND_SERIAL)
      if (mass < 0)
        throw std::invalid_argument("Negative mass given as argument");
#endif
      _m = mass;
      return *this;
    }

    /// Set the p coordinates and mass simultaneously
    PHIRST_HOST_DEVICE
    P4& setPM(double px, double py, double pz, double mass) {
#if defined(PHIRST_BACKEND_SERIAL)
      if (mass < 0)
        throw std::invalid_argument("Negative mass given as argument");
#endif
      setPx(px); setPy(py); setPz(pz);
      setM(mass);
      return *this;
    }
    /// Alias for setPM
    PHIRST_HOST_DEVICE
    P4& setXYZM(double px, double py, double pz, double mass) {
      return setPM(px, py, pz, mass);
    }

    /// Set the p coordinates and energy simultaneously
    /// @warning For numerical stability, prefer setPM when possible
    PHIRST_HOST_DEVICE
    P4& setPE(double px, double py, double pz, double E) {
#if defined(PHIRST_BACKEND_SERIAL)
      if (E < 0)
        throw std::invalid_argument("Negative energy given as argument");
#endif
      setPx(px); setPy(py); setPz(pz);
      const double mass2 = sqr(E) - sqr(p());
      setM(mass2 >= 0.0 ? phirst::math::sqrt(mass2) : 0.0);
      return *this;
    }
    /// Alias for setPE
    PHIRST_HOST_DEVICE
    P4& setXYZE(double px, double py, double pz, double E) {
      return setPE(px, py, pz, E);
    }

    /// Set the vector state from (eta,phi,energy) coordinates and the mass
    ///
    /// eta = -ln(tan(theta/2))
    /// -> theta = 2 atan(exp(-eta))
    PHIRST_HOST_DEVICE
    P4& setEtaPhiME(double eta, double phi, double mass, double E) {
#if defined(PHIRST_BACKEND_SERIAL)
      if (mass < 0)
        throw std::invalid_argument("Negative mass given as argument");
      if (E < 0)
        throw std::invalid_argument("Negative energy given as argument");
#endif
      const double theta = 2 * phirst::math::atan(phirst::math::exp(-eta));
#if defined(PHIRST_BACKEND_SERIAL)
      if (theta < 0 || theta > phirst::math::pi)
        throw std::domain_error("Polar angle outside 0..pi in calculation");
#endif
      setThetaPhiME(theta, phi, mass, E);
      return *this;
    }

    /// Set the vector state from (eta,phi,pT) coordinates and the mass
    ///
    /// eta = -ln(tan(theta/2))
    /// -> theta = 2 atan(exp(-eta))
    PHIRST_HOST_DEVICE
    P4& setEtaPhiMPt(double eta, double phi, double mass, double pt) {
#if defined(PHIRST_BACKEND_SERIAL)
      if (mass < 0)
        throw std::invalid_argument("Negative mass given as argument");
      if (pt < 0)
        throw std::invalid_argument("Negative transverse momentum given as argument");
#endif
      const double theta = 2 * phirst::math::atan(phirst::math::exp(-eta));
#if defined(PHIRST_BACKEND_SERIAL)
      if (theta < 0 || theta > phirst::math::pi)
        throw std::domain_error("Polar angle outside 0..pi in calculation");
#endif
      const double p = pt / phirst::math::sin(theta);
      const double E = phirst::math::sqrt( sqr(p) + sqr(mass) );
      setThetaPhiME(theta, phi, mass, E);
      return *this;
    }

    /// Set the vector state from (y,phi,energy) coordinates and the mass
    ///
    /// y = 0.5 * ln((E+pz)/(E-pz))
    /// -> (E^2 - pz^2) exp(2y) = (E+pz)^2
    ///  & (E^2 - pz^2) exp(-2y) = (E-pz)^2
    /// -> E = sqrt(pt^2 + m^2) cosh(y)
    /// -> pz = sqrt(pt^2 + m^2) sinh(y)
    /// -> sqrt(pt^2 + m^2) = E / cosh(y)
    PHIRST_HOST_DEVICE
    P4& setRapPhiME(double y, double phi, double mass, double E) {
#if defined(PHIRST_BACKEND_SERIAL)
      if (mass < 0)
        throw std::invalid_argument("Negative mass given as argument");
      if (E < 0)
        throw std::invalid_argument("Negative energy given as argument");
#endif
      const double sqrt_pt2_m2 = E / phirst::math::cosh(y);
      const double pt = phirst::math::sqrt( sqr(sqrt_pt2_m2) - sqr(mass) );
#if defined(PHIRST_BACKEND_SERIAL)
      if (pt < 0)
        throw std::domain_error("Negative transverse momentum in calculation");
#endif
      const double pz = sqrt_pt2_m2 * phirst::math::sinh(y);
      const double px = pt * phirst::math::cos(phi);
      const double py = pt * phirst::math::sin(phi);
      setPM(px, py, pz, mass);
      return *this;
    }

    /// Set the vector state from (y,phi,pT) coordinates and the mass
    ///
    /// y = 0.5 * ln((E+pz)/(E-pz))
    /// -> E = sqrt(pt^2 + m^2) cosh(y)  [see above]
    PHIRST_HOST_DEVICE
    P4& setRapPhiMPt(double y, double phi, double mass, double pt) {
#if defined(PHIRST_BACKEND_SERIAL)
      if (mass < 0)
        throw std::invalid_argument("Negative mass given as argument");
      if (pt < 0)
        throw std::invalid_argument("Negative transverse mass given as argument");
#endif
      const double E = phirst::math::sqrt( sqr(pt) + sqr(mass) ) * phirst::math::cosh(y);
#if defined(PHIRST_BACKEND_SERIAL)
      if (E < 0)
        throw std::domain_error("Negative energy in calculation");
#endif
      setRapPhiME(y, phi, mass, E);
      return *this;
    }

    /// Set the vector state from (theta,phi,energy) coordinates and the mass
    ///
    /// p = sqrt(E^2 - mass^2)
    /// pz = p cos(theta)
    /// pt = p sin(theta)
    PHIRST_HOST_DEVICE
    P4& setThetaPhiME(double theta, double phi, double mass, double E) {
#if defined(PHIRST_BACKEND_SERIAL)
      if (theta < 0 || theta > phirst::math::pi)
        throw std::invalid_argument("Polar angle outside 0..pi given as argument");
      if (mass < 0)
        throw std::invalid_argument("Negative mass given as argument");
      if (E < 0)
        throw std::invalid_argument("Negative energy given as argument");
#endif
      const double p = phirst::math::sqrt( sqr(E) - sqr(mass) );
      const double pz = p * phirst::math::cos(theta);
      const double pt = p * phirst::math::sin(theta);
#if defined(PHIRST_BACKEND_SERIAL)
      if (pt < 0)
        throw std::invalid_argument("Negative transverse momentum in calculation");
#endif
      const double px = pt * phirst::math::cos(phi);
      const double py = pt * phirst::math::sin(phi);
      setPM(px, py, pz, mass);
      return *this;
    }

    /// Set the vector state from (theta,phi,pT) coordinates and the mass
    ///
    /// p = pt / sin(theta)
    /// pz = p cos(theta)
    /// E = sqrt(p^2 + mass^2)
    PHIRST_HOST_DEVICE
    P4& setThetaPhiMPt(double theta, double phi, double mass, double pt) {
#if defined(PHIRST_BACKEND_SERIAL)
      if (theta < 0 || theta > phirst::math::pi)
        throw std::invalid_argument("Polar angle outside 0..pi given as argument");
      if (mass < 0)
        throw std::invalid_argument("Negative mass given as argument");
      if (pt < 0)
        throw std::invalid_argument("Negative transverse momentum given as argument");
#endif
      const double p = pt / phirst::math::sin(theta);
      const double px = pt * phirst::math::cos(phi);
      const double py = pt * phirst::math::sin(phi);
      const double pz = p * phirst::math::cos(theta);
      //const double E = sqrt( sqr(p) + sqr(mass) );
      setPM(px, py, pz, mass);
      return *this;
    }

    /// Set the vector state from (pT,phi,energy) coordinates and the mass
    ///
    /// pz = sqrt(E^2 - mass^2 - pt^2)
    PHIRST_HOST_DEVICE
    P4& setPtPhiME(double pt, double phi, double mass, double E) {
#if defined(PHIRST_BACKEND_SERIAL)
      if (pt < 0)
        throw std::invalid_argument("Negative transverse momentum given as argument");
      if (mass < 0)
        throw std::invalid_argument("Negative mass given as argument");
      if (E < 0)
        throw std::invalid_argument("Negative energy given as argument");
#endif
      const double px = pt * phirst::math::cos(phi);
      const double py = pt * phirst::math::sin(phi);
      const double pz = phirst::math::sqrt(sqr(E) - sqr(mass) - sqr(pt));
      setPM(px, py, pz, mass);
      return *this;
    }

    //@}


    /// @name Coordinate getters
    //@{

    /// Get px^2
    PHIRST_HOST_DEVICE double px2() const { return sqr(_x); }
    /// Get px
    PHIRST_HOST_DEVICE double px() const { return _x; }
    /// Get py^2
    PHIRST_HOST_DEVICE double py2() const { return sqr(_y); }
    /// Get py
    PHIRST_HOST_DEVICE double py() const { return _y; }
    /// Get pz^2
    PHIRST_HOST_DEVICE double pz2() const { return sqr(_z); }
    /// Get pz
    PHIRST_HOST_DEVICE double pz() const { return _z; }
    /// Get m^2
    PHIRST_HOST_DEVICE double m2() const { return sqr(_m); }
    /// Get m
    PHIRST_HOST_DEVICE double m() const { return _m; }

    /// Get E^2
    PHIRST_HOST_DEVICE double E2() const { return p2() + sqr(_m); }
    /// Get E
    PHIRST_HOST_DEVICE double E() const { return phirst::math::sqrt(E2()); }
    /// Get the spatial 3-vector |p|^2
    PHIRST_HOST_DEVICE double p2() const { return sqr(px()) + sqr(py()) + sqr(pz()); }
    /// Get the spatial 3-vector |p|
    PHIRST_HOST_DEVICE double p() const { return phirst::math::sqrt(p2()); }
    /// Get the spatial 3-vector |px^2 + py^2|
    PHIRST_HOST_DEVICE double rho2() const { return sqr(px()) + sqr(py()); }
    /// Get the spatial 3-vector sqrt|px^2 + py^2|
    PHIRST_HOST_DEVICE double rho() const { return phirst::math::sqrt(rho2()); }
    /// Get the transverse momentum squared (same as rho2)
    PHIRST_HOST_DEVICE double pT2() const { return rho2(); }
    /// Get the transverse momentum (same as rho)
    PHIRST_HOST_DEVICE double pT() const { return rho(); }

    /// Get the spatial phi (in the range -pi .. pi)
    PHIRST_HOST_DEVICE double phi() const { if (rho2() == 0) return 0; else return phirst::math::atan2(py(), px()); }
    /// Get the spatial phi (in the range 0 .. 2pi)
    PHIRST_HOST_DEVICE double phi_02pi() const { if (rho2() == 0) return 0; else return phi() + phirst::math::pi; }

    /// Get the spatial theta (in the range 0 .. pi)
    PHIRST_HOST_DEVICE double theta() const { if (p2() == 0) return 0; else
	if (pz() == 0) return phirst::math::halfPi; else return phirst::math::atan2(rho(), pz()); } //< atan2(+ve, z) is +ve
    /// Get the spatial-vector pseudorapidity
    PHIRST_HOST_DEVICE double eta() const { return phirst::math::copysign(phirst::math::log((p() + phirst::math::fabs(pz())) / pT()), pz()); }
    /// Get the spatial-vector absolute pseudorapidity
    PHIRST_HOST_DEVICE double abseta() const { return phirst::math::fabs(eta()); }
    /// Get the 4-momentum rapidity
    PHIRST_HOST_DEVICE double rap() const { return 0.5 * phirst::math::log((E() + pz()) / (E() - pz())); }
    /// Get the 4-momentum absolute rapidity
    PHIRST_HOST_DEVICE double absrap() const { return phirst::math::fabs(rap()); }

    //@}


    /// @name Calculations w.r.t. other P4 vectors
    //@{

    /// Spatial dot product
    PHIRST_HOST_DEVICE double dot3(const P4& v) const { return px()*v.px() + py()*v.py() + pz()*v.pz(); }
    /// Lorentz dot product with the positive metric term on E
    PHIRST_HOST_DEVICE double dot(const P4& v) const { return E()*v.E() - dot3(v); }
    /// Spatial angle to another P4 vector
    PHIRST_HOST_DEVICE double angleTo(const P4& v) const { return phirst::math::acos( dot3(v) /p()/v.p() ); }
    /// Difference in phi between two vectors
    PHIRST_HOST_DEVICE double deltaPhi(const P4& v) const { return deltaphi(phi(), v.phi()); }
    /// Difference in pseudorapidity between two vectors
    PHIRST_HOST_DEVICE double deltaEta(const P4& v) const { return phirst::math::fabs(eta() - v.eta()); }
    /// Difference in rapidity between two vectors
    PHIRST_HOST_DEVICE double deltaRap(const P4& v) const { return phirst::math::fabs(rap() - v.rap()); }
    /// Difference in pseudorapidity-based R^2 between two vectors
    PHIRST_HOST_DEVICE double deltaR2_eta(const P4& v) const { return sqr(deltaEta(v)) + sqr(deltaPhi(v)); }
    /// Difference in pseudorapidity-based R between two vectors
    PHIRST_HOST_DEVICE double deltaR_eta(const P4& v) const { return phirst::math::sqrt(deltaR2_eta(v)); }
    /// Difference in rapidity-based R^2 between two vectors
    PHIRST_HOST_DEVICE double deltaR2_rap(const P4& v) const { return sqr(deltaRap(v)) + sqr(deltaPhi(v)); }
    /// Difference in rapidity-based R between two vectors
    PHIRST_HOST_DEVICE double deltaR_rap(const P4& v) const { return phirst::math::sqrt(deltaR2_rap(v)); }

    //@}

    /// @name Self-modifying operators
    //@{
    PHIRST_HOST_DEVICE P4  operator - () const { P4 rtn; return rtn.setPM(-_x, -_y, -_z, _m); } //< Not self-modifying...
    PHIRST_HOST_DEVICE P4& operator += (const P4& v) { double e = E() + v.E(); _x += v.px(); _y += v.py(); _z += v.pz(); const double m2 = sqr(e) - p2(); _m = m2 >= 0.0 ? phirst::math::sqrt(m2) : 0.0; return *this; }
    PHIRST_HOST_DEVICE P4& operator -= (const P4& v) { double e = E() - v.E(); _x -= v.px(); _y -= v.py(); _z -= v.pz(); const double m2 = sqr(e) - p2(); _m = m2 >= 0.0 ? phirst::math::sqrt(m2) : 0.0; return *this; }
    PHIRST_HOST_DEVICE P4& operator *= (double a) { _x *= a; _y *= a; _z *= a; _m *= a; return *this; }
    PHIRST_HOST_DEVICE P4& operator /= (double a) { _x /= a; _y /= a; _z /= a; _m /= a; return *this; }
    //@}

  };


  /// @name String representations
  //@{

#if defined(PHIRST_BACKEND_SERIAL)
  /// Make a string representation of the vector
  inline std::string to_str(const P4& p4) {
    std::stringstream ss;
    ss << "(" << p4.px() << ", " << p4.py() << ", " << p4.pz() << "; " << p4.E() << ")";
    return ss.str();
  }

  /// Write a string representation of the vector to the provided stream
  inline std::ostream& operator <<(std::ostream& ostr, const P4& p4) {
    ostr << to_str(p4);
    return ostr;
  }
#endif

  //@}


  /// Convenience "external" functions
  //@{

  /// Lorentz inner product between two vectors
  PHIRST_INLINE double dot(const P4& a, const P4& b) {
    return a.dot(b);
  }


  /// Angle between two vectors
  PHIRST_INLINE double angle(const P4& a, const P4& b) {
    return a.angleTo(b);
  }


  /// Difference in phi between two vectors
  PHIRST_INLINE double deltaPhi(const P4& a, const P4& b) {
    return a.deltaPhi(b);
  }


  /// Difference in pseudorapidity between two vectors
  PHIRST_INLINE double deltaEta(const P4& a, const P4& b) {
    return a.deltaEta(b);
  }


  /// Difference in rapidity between two vectors
  PHIRST_INLINE double deltaRap(const P4& a, const P4& b) {
    return a.deltaRap(b);
  }


  /// Difference in pseudorapidity-based R^2 between two vectors
  PHIRST_INLINE double deltaR2_eta(const P4& a, const P4& b) {
    return a.deltaR2_eta(b);
  }
  /// Difference in pseudorapidity-based R between two vectors
  PHIRST_INLINE double deltaR_eta(const P4& a, const P4& b) {
    return a.deltaR_eta(b);
  }


  /// Difference in rapidity-based R^2 between two vectors
  PHIRST_INLINE double deltaR2_rap(const P4& a, const P4& b) {
    return a.deltaR2_rap(b);
  }
  /// Difference in rapidity-based R between two vectors
  PHIRST_INLINE double deltaR_rap(const P4& a, const P4& b) {
    return a.deltaR_rap(b);
  }

  //@}


  /// @name Operators taking two vectors
  //@{
  PHIRST_INLINE P4 operator + (const P4& a, const P4& b) { P4 rtn = a; return rtn += b; }
  PHIRST_INLINE P4 operator - (const P4& a, const P4& b) { P4 rtn = a; return rtn -= b; }
  PHIRST_INLINE P4 operator * (const P4& a, double f) { P4 rtn = a; return rtn *= f; }
  PHIRST_INLINE P4 operator * (double f, const P4& a) { P4 rtn = a; return rtn *= f; }
  PHIRST_INLINE P4 operator / (const P4& a, double f) { P4 rtn = a; return rtn /= f; }
  //@}


  /// Function/functor for container<const P4> sorting (cf. std::less)
  template <typename T>
  inline bool _cmpPtDesc(const T& a, const T& b) {
    return a.pT2() >= b.pT2();
  }

  /// Function/functor for container<const P4*> sorting (cf. std::less)
  template <typename T>
  inline bool _cmpPtDescPtr(const T* a, const T* b) {
     return _cmpPtDesc(*a, *b);
  }


}
