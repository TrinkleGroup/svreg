/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   see LLNL copyright notice at bottom of file
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(spline/tree,PairSplineTree)

#else

#ifndef LMP_PAIR_MEAM_SPLINE_H
#define LMP_PAIR_MEAM_SPLINE_H

#include "pair.h"
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <functional>
#include <map>
#include <utility>

namespace LAMMPS_NS {

#define SPLINE_MEAM_SUPPORT_NON_GRID_SPLINES 0
typedef std::vector<double> dvec;
typedef std::vector<std::vector<dvec>> dvec3;
// typedef double ***dvec3;

class PairSplineTree : public Pair
{

// Helper functions for splitting a string on a delimiter
template <typename Out>
static void split1(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

static std::vector<std::string> helper_split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split1(s, delim, std::back_inserter(elems));
    return elems;
}

// 'trim' and 'reduce' code from SO post:
// https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string

static std::string helper_trim(const std::string& str,
                 const std::string& whitespace = " \t\n")
{
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

static std::string helper_reduce(const std::string& str,
                   const std::string& fill = " ",
                   const std::string& whitespace = " \t")
{
    // trim first
    auto result = helper_trim(str, whitespace);

    // replace sub ranges
    auto beginSpace = result.find_first_of(whitespace);
    while (beginSpace != std::string::npos)
    {
        const auto endSpace = result.find_first_not_of(whitespace, beginSpace);
        const auto range = endSpace - beginSpace;

        result.replace(beginSpace, range, fill);

        const auto newStart = beginSpace + fill.length();
        beginSpace = result.find_first_of(whitespace, newStart);
    }

    return result;
}

static int helper_pairing(int a, int b) {
  // Cantor pairing function; maps pairs of indices to a unique integer
  // Note: sort x and y to return a unique value regardless of order
  int x, y;

  if (a < b) {
    x = a;
    y = b;
  } else {
    x = b;
    y = a;
  }

  return (x*x + 2*x*y + y*y + 3*x + y)/2;
}

static int helper_pairing(int a){return a;}

// TODO: If a 4-body SV is ever defined, we'll need a helper_pairing(a, b, c)

public:
  PairSplineTree(class LAMMPS *);
  virtual ~PairSplineTree();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void get_coeff(double *, double *);
  double pair_density(int );
  double three_body_density(int );
  void init_style();
  void init_list(int, class NeighList *);
  double init_one(int, int);

  // helper functions for compute()

  int ij_to_potl(const int itype, const int jtype, const int ntypes) const {
    return  jtype - 1 + (itype-1)*ntypes - (itype-1)*itype/2;
  }
  int i_to_potl(const int itype) const { return itype-1; }


  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();

protected:
  char **elements;              // names of unique elements
  int *map;                     // mapping from atom types to elements
  int nelements;                // # of unique elements

  class StructureVector {
    public:
      StructureVector() :
        svType(""),
        name(""),
        neighbor_elements(),
        cutoffs({0, 0}),
        num_knots(),
        parameters(),
        numKnots(0),
        numParams(0),
        g() {}

      void parse(FILE* fp, Error* error);

      class SplineFunction {
      public:
        /// Default constructor.
        SplineFunction() : X(NULL), Xs(NULL), Y(NULL), Y2(NULL), Ydelta(NULL), N(0) {}

        /// Destructor.
        ~SplineFunction() {
          delete[] X;
          delete[] Xs;
          delete[] Y;
          delete[] Y2;
          delete[] Ydelta;
        }

        /// Initialization of spline function.
        void init(int _N, double inX[], dvec parameters) {
          N = _N;

          deriv0 = parameters[N];
          derivN = parameters[N+1];

          X  = new double[N];
          Xs = new double[N];
          Y = new double[N];

          double xmin = X[0];
          for(int ii=0; ii<N; ii++) {
            X[ii] = inX[ii];
            Xs[ii] = inX[ii] - xmin;
            Y[ii] = parameters[ii];
          }

          Y2 = new double[N];
          Ydelta = new double[N];
        }

        /// Adds a knot to the spline.
        void setKnot(int n, double x, double y) { X[n] = x; Y[n] = y; }

        /// Returns the number of knots.
        int numKnots() const { return N; }

        /// Calculates the second derivatives of the cubic spline.
        void prepareSpline(Error* error);

        /// Evaluates the spline function at position x.
        inline double eval(double x) const
        {
          x -= xmin;
          if(x <= 0.0) {  // Left extrapolation.
            return Y[0] + deriv0 * x;
          }
          else if(x >= xmax_shifted) {  // Right extrapolation.
            return Y[N-1] + derivN * (x - xmax_shifted);
          }
          else {
            // For a spline with regular grid, we directly calculate the interval X is in.
            int klo = (int)(x*inv_h);
            int khi = klo + 1;
            double a = Xs[khi] - x;
            double b = h - a;
            return Y[khi] - a * Ydelta[klo] +
              ((a*a - hsq) * a * Y2[klo] + (b*b - hsq) * b * Y2[khi]);
          }
        }

        /// Evaluates the spline function and its first derivative at position x.
        inline double eval(double x, double& deriv) const
        {
          x -= xmin;
          if(x <= 0.0) {  // Left extrapolation.
            deriv = deriv0;
            return Y[0] + deriv0 * x;
          }
          else if(x >= xmax_shifted) {  // Right extrapolation.
            deriv = derivN;
            return Y[N-1] + derivN * (x - xmax_shifted);
          }
          else {
            // For a spline with regular grid, we directly calculate the interval X is in.
            int klo = (int)(x*inv_h);
            int khi = klo + 1;
            double a = Xs[khi] - x;
            double b = h - a;
            deriv = Ydelta[klo] + ((3.0*b*b - hsq) * Y2[khi]
                                  - (3.0*a*a - hsq) * Y2[klo]);
            return Y[khi] - a * Ydelta[klo] +
              ((a*a - hsq) * a * Y2[klo] + (b*b - hsq) * b * Y2[khi]);
          }
        }

        /// Returns the number of bytes used by this function object.
        double memory_usage() const { return sizeof(*this) + sizeof(X[0]) * N * 3; }

        /// Returns the cutoff radius of this function.
        double cutoff() const { return X[N-1]; }

        /// Writes a Gnuplot script that plots the spline function.
        void writeGnuplot(const char* filename, const char* title = NULL) const;

        /// Broadcasts the spline function parameters to all processors.
        void communicate(MPI_Comm& world, int me);

      private:
        double* X;       // Positions of spline knots
        double* Xs;      // Shifted positions of spline knots
        double* Y;       // Function values at spline knots
        double* Y2;      // Second derivatives at spline knots
        double* Ydelta;  // If this is a grid spline, Ydelta[i] = (Y[i+1]-Y[i])/h
        int N;           // Number of spline knots
        double deriv0;   // First derivative at knot 0
        double derivN;   // First derivative at knot (N-1)
        double xmin;     // The beginning of the interval on which the spline function is defined.
        double xmax;     // The end of the interval on which the spline function is defined.
        int isGridSpline;// Indicates that all spline knots are on a regular grid.
        double h;        // The distance between knots if this is a grid spline with equidistant knots.
        double hsq;      // The squared distance between knots if this is a grid spline with equidistant knots.
        double inv_h;    // (1/h), used to avoid numerical errors in binnning for grid spline with equidistant knots.
        double xmax_shifted; // The end of the spline interval after it has been shifted to begin at X=0.
      };

      std::string svType;
      std::string name;
      std::vector<std::string> neighbor_elements;
      std::vector<int> neighbor_types;
      double cutoffs[2];
      // std::vector<std::string> components;
      std::vector<int> num_knots;
      dvec parameters;
      int numKnots;
      int numParams;
      std::vector<SplineFunction *> splines;
      int bondType;
      SplineFunction * g;

      void display();
      void splineSetup(Error * error);

  };

  class Node {
    public:
      Node() : description(""), energies(NULL), forces(NULL), arity(-1) {}
      std::string description;
      dvec energies;
      dvec3 forces;
      int arity;

      // std::vector<std::vector<int>> force_tags;  // Used for indexing forces properly

      std::function<dvec (dvec)> f1;
      std::function<dvec (dvec, dvec)> f2;

      std::function<dvec3 (std::pair<dvec, dvec3>)> d1;
      std::function<
        dvec3 (
          std::pair<dvec, dvec3>,
          std::pair<dvec, dvec3>
        )
      > d2;

      dvec eval(std::pair<dvec, dvec3> inp) {
        // Arity=1 function evaluation
        return f1(inp.first);
      };

      dvec eval(std::pair<dvec, dvec3> inp1, std::pair<dvec, dvec3> inp2) {
        // Arity=2 function evaluation
        return f2(inp1.first, inp2.first);
      };

      dvec3 deriv(std::pair<dvec, dvec3> inp) {
        return d1(inp);
      }

      dvec3 deriv(std::pair<dvec, dvec3> inp1, std::pair<dvec, dvec3> inp2) {
        return d2(inp1, inp2);
      }
  };
 
  /*
  Function nodes are designed to work with 2-tuple inputs, which helps with
  writing the code for calculating derivatives.

  The first entry in the tuple is the function value (energy), while the second
  entry is the function derivative (forces).

  N = number of atoms
  Energies should be per-atom vector with a shape of (N,)
  Forces should be a 2D per-atom vector with a shape of (N, N, 3)
  */

  static dvec add(dvec x, dvec y) {
    int N = x.size();
    dvec energies(N, 0);

    for (int i=0; i<N; i++) {
      energies[i] = x[i]+y[i];
    }

    return energies;
  }

  /*
  TODO: trying to figure out how to get forces working properly
  */

  static dvec3 _deriv_add(std::pair<dvec, dvec3> inp1, std::pair<dvec, dvec3> inp2) {
    int N = inp1.first.size();
    dvec3 forces;

    // Initialize forces array
    for (int i=0; i<N; i++) {
      std::vector<dvec> tmp;

      for (int j=0; j<N; j++) {
        tmp.push_back({0, 0, 0});
      }

      forces.push_back(tmp);
    }

    // Fill forces array
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
        for (int k=0; k<3; k++) {
          forces[i][j][k] = inp1.second[i][j][k] + inp2.second[i][j][k];
        }
      }
    }

    return forces;
  }


  static dvec mul(dvec x, dvec y) {
    int N = x.size();
    dvec energies(N, 0);

    for (int i=0; i<N; i++) {
      energies[i] = x[i]*y[i];
    }

    return energies;
  }

  static dvec3 _deriv_mul(std::pair<dvec, dvec3> inp1, std::pair<dvec, dvec3> inp2) {

    int N = inp1.first.size();
    dvec3 forces;

    // Initialize forces array
    for (int i=0; i<N; i++) {
      std::vector<dvec> tmp;

      for (int j=0; j<N; j++) {
        tmp.push_back({0, 0, 0});
      }

      forces.push_back(tmp);
    }

    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
        for (int k=0; k<3;k ++) {
          forces[i][j][k] = inp1.first[i]*inp2.second[i][j][k] + inp2.first[i]*inp1.second[i][j][k];
        }
      }
    }

    return forces;
  }

  std::map<
    std::string,  // key
    std::pair<
      std::function<dvec (dvec)>,  // arity=1 function
      std::function<dvec3 (std::pair<dvec, dvec3>)>>  // arity=1 deriv
  > arity1Functions;

  std::map<
    std::string,  // key
    std::pair<
      std::function<dvec (dvec, dvec)>,  //arity=2 function
      std::function<dvec3 (  // arity=2 deriv
        std::pair<dvec, dvec3>,
        std::pair<dvec, dvec3>
      )>>
  > arity2Functions;

  class SVNode : public StructureVector, public Node {
    public:
      int hostType;
      std::string hostElement;
  };

  int totalNumParams;  // numKnots + 2 (boundary conditions) for all SVNodes
  std::map<int, std::vector<Node *>> nodes;  // Includes function nodes
  std::vector<SVNode *> svnodes;  // Useful pointer to all SVNodes
  std::map<int, std::vector<SVNode *>> rho_nodes;  // key=host, val=Rho SVNs
  std::map<int, std::vector<SVNode *>> ffg_nodes;  // key=host, val=FFG SVNs

  /// Helper data structure for potential routine.
  struct MEAM2Body {
    int tag;  // holds the (local) index of the second atom (j)
    int globalTag;  // holds the _global_ index of atom j
    double r;
    dvec f, fprime;
    double del[3];
  };

  std::map<std::string, StructureVector> sv_templates;

  double cutoff;          // The cutoff radius

  int nmax;               // Size of temporary array.
  int maxNeighbors;       // The last maximum number of neighbors a single atoms has.
  MEAM2Body* twoBodyInfo; // Temporary array.

  void read_file(char **arg, Error * error);
  void allocate();


};
}

#endif
#endif

/* ----------------------------------------------------------------------
 * Spline-based Modified Embedded Atom method (MEAM) potential routine.
 *
 * Copyright (2011) Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Alexander Stukowski (<alex@stukowski.com>).
 * LLNL-CODE-525797 All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2, dated June 1991.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * Our Preamble Notice
 * A. This notice is required to be provided under our contract with the
 * U.S. Department of Energy (DOE). This work was produced at the
 * Lawrence Livermore National Laboratory under Contract No.
 * DE-AC52-07NA27344 with the DOE.
 *
 * B. Neither the United States Government nor Lawrence Livermore National
 * Security, LLC nor any of their employees, makes any warranty, express or
 * implied, or assumes any liability or responsibility for the accuracy,
 * completeness, or usefulness of any information, apparatus, product, or
 * process disclosed, or represents that its use would not infringe
 * privately-owned rights.
 *
 * C. Also, reference herein to any specific commercial products, process,
 * or services by trade name, trademark, manufacturer or otherwise does not
 * necessarily constitute or imply its endorsement, recommendation, or
 * favoring by the United States Government or Lawrence Livermore National
 * Security, LLC. The views and opinions of authors expressed herein do not
 * necessarily state or reflect those of the United States Government or
 * Lawrence Livermore National Security, LLC, and shall not be used for
 * advertising or product endorsement purposes.
 *
 * See file 'pair_spline_meam.cpp' for history of changes.
------------------------------------------------------------------------- */


