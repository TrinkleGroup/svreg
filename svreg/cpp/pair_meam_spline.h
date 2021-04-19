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

namespace LAMMPS_NS {

#define SPLINE_MEAM_SUPPORT_NON_GRID_SPLINES 0

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
        name(""),
        neighbor_elements(),
        cutoffs({0, 0}),
        // components(),
        num_knots(),
        parameters(),
        numKnots(0),
        numParams(0) {}

      void parse(FILE* fp, Error* error);

      class SplineFunction {
      public:
        /// Default constructor.
        SplineFunction() : X(NULL), Xs(NULL), Y(NULL), Y2(NULL), Ydelta(NULL), N(0) {}

        // FunctionNode(std::function<double (double, double)> func) :

        /// Destructor.
        ~SplineFunction() {
          delete[] X;
          delete[] Xs;
          delete[] Y;
          delete[] Y2;
          delete[] Ydelta;
        }

        /// Initialization of spline function.
        void init(int _N, double inX[], std::vector<double> parameters) {
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

      std::string name;
      std::vector<std::string> neighbor_elements;
      double cutoffs[2];
      // std::vector<std::string> components;
      std::vector<int> num_knots;
      std::vector<double> parameters;
      int numKnots;
      int numParams;
      std::vector<SplineFunction *> splines;
      int bondType;

      void display();
      void splineSetup(Error * error);

  };

  static double add(double x, double y) {return x+y;}

  class Node {
    public:
      Node() : description("") {}
      std::string description;
  };

  class FunctionNode : public Node {
    public:
      FunctionNode() : f(NULL) {}
      FunctionNode(std::function<double (double, double)> func) :
        f(func) {}

      std::function<double (double, double)> f;
  };


  class SVNode : public StructureVector, public Node {};

  int totalNumParams;
  std::map<std::string, std::vector<Node *>> nodes;
  std::vector<SVNode *> svnodes;
  std::vector<SVNode *> rho_nodes;
  std::vector<SVNode *> ffg_nodes;

  /// Helper data structure for potential routine.
  struct MEAM2Body {
    int tag;  // holds the index of the second atom (j)
    double r;
    double f, fprime;
    double del[3];
  };

  std::map<std::string, StructureVector> sv_templates;

  // TODO: build the map svs[elem][svname] = StructureVector()
  // Could be svs[elem][svname] = idx, where idx indexes rho_svs or ffg_svs
  // Need something like svs[host][neighbors]
  // Or maybe svs[host][ij_to_potl(itype,jtype)]?
  // Could define a bondType object: i+j?

  double* zero_atom_energies; // Shift embedding energy by this value to make it zero for a single atom in vacuum.

  double cutoff;          // The cutoff radius
  bool threeBody;  // flag for running triplet loop

  int nmax;               // Size of temporary array.
  int maxNeighbors;       // The last maximum number of neighbors a single atoms has.
  MEAM2Body* twoBodyInfo; // Temporary array.

  double* Uprime_values;

  void read_file(const char* filename);
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


