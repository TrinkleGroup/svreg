/* ----------------------------------------------------------------------
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
   Contributing author: Alexander Stukowski (LLNL), alex@stukowski.com
                        Will Tipton (Cornell), wwt26@cornell.edu
                        Dallas R. Trinkle (UIUC), dtrinkle@illinois.edu
                        Pinchao Zhang (UIUC)
   see LLNL copyright notice at bottom of file
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 * File history of changes:
 * 25-Oct-10 - AS: First code version.
 * 17-Feb-11 - AS: Several optimizations (introduced MEAM2Body struct).
 * 25-Mar-11 - AS: Fixed calculation of per-atom virial stress.
 * 11-Apr-11 - AS: Adapted code to new memory management of LAMMPS.
 * 24-Sep-11 - AS: Adapted code to new interface of Error::one() function.
 * 20-Jun-13 - WT: Added support for multiple species types
 * 25-Apr-17 - DRT/PZ: Modified format of multiple species type to
                       conform with pairing, updated to LAMMPS style
------------------------------------------------------------------------- */

#include "pair_meam_spline.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSplineTree::PairSplineTree(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;

  nelements = 0;
  elements = NULL;

  nmax = 0;
  maxNeighbors = 0;
  twoBodyInfo = NULL;

  comm_forward = 1;
  comm_reverse = 0;

  totalNumParams = 0;
  
  arity1Functions.insert(
    std::make_pair(
      "global",
      std::make_pair(PairSplineTree::global_state, PairSplineTree::_deriv_global_state)
    )
  );
 
  arity1Functions.insert(
    std::make_pair(
      "softplus",
      std::make_pair(PairSplineTree::splus, PairSplineTree::_deriv_splus)
    )
  );
 
  arity2Functions.insert(
    std::make_pair(
      "add",
      std::make_pair(PairSplineTree::add, PairSplineTree::_deriv_add)
    )
  );

  arity2Functions.insert(
    std::make_pair(
      "mul",
      std::make_pair(PairSplineTree::mul, PairSplineTree::_deriv_mul)
    )
  );

}

/* ---------------------------------------------------------------------- */

PairSplineTree::~PairSplineTree()
{
  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;

  delete[] twoBodyInfo;

  if(allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    delete [] map;
  }
}

/* ---------------------------------------------------------------------- */

void PairSplineTree::compute(int eflag, int vflag)
{
  const double* const * const x = atom->x;
  double* const * const forces = atom->f;
  const int ntypes = atom->ntypes;

  ev_init(eflag, vflag);

  // Grow per-atom array if necessary

  for (int itype=0; itype<rho_nodes.size(); itype++) {
    for (SVNode * rho : rho_nodes[itype]) {
      rho->energies.resize(listfull->inum);

      // Initialize forces array
      for (int i=0; i<listfull->inum; i++) {
        std::vector<dvec> tmp;

        for (int j=0; j<listfull->inum; j++) {
          tmp.push_back({0, 0, 0});
        }

        rho->forces.push_back(tmp);
      }
    }
  }

  for (int itype=0; itype<ffg_nodes.size(); itype++) {
    for (SVNode * ffg : ffg_nodes[itype]) {
      ffg->energies.resize(listfull->inum);

      // Initialize forces array
      for (int i=0; i<listfull->inum; i++) {
        std::vector<dvec> tmp;

        for (int j=0; j<listfull->inum; j++) {
          tmp.push_back({0, 0, 0});
        }

        ffg->forces.push_back(tmp);
      }
    }
  }

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
  }

  // Determine the maximum number of neighbors a single atom has

  int newMaxNeighbors = 0;
  for(int ii = 0; ii < listfull->inum; ii++) {
    int jnum = listfull->numneigh[listfull->ilist[ii]];
    if(jnum > newMaxNeighbors)
      newMaxNeighbors = jnum;
  }

  // Allocate array for temporary bond info

  if(newMaxNeighbors > maxNeighbors) {
    maxNeighbors = newMaxNeighbors;
    delete[] twoBodyInfo;
    twoBodyInfo = new MEAM2Body[maxNeighbors];
  }

  atom->map_init();
  atom->map_set();

  // Loop over all atoms
  for(int ii = 0; ii < listfull->inum; ii++) {
    int i = listfull->ilist[ii];
    int iGlobalTag = atom->tag[ii]-1;
    int numBonds = 0;

    MEAM2Body* nextTwoBodyInfo = twoBodyInfo;
    double rho_value = 0;
    const int ntypes = atom->ntypes;
    const int itype = atom->type[i];

    int num_ffg = ffg_nodes[itype-1].size();

    double per_atom_energy = 0;

    // Loop over all neighbors (accounts for i->j and j->i)
    for(int jj = 0; jj < listfull->numneigh[i]; jj++) {
      int j = listfull->firstneigh[i][jj];
      j &= NEIGHMASK;

      double jdelx = x[j][0] - x[i][0];
      double jdely = x[j][1] - x[i][1];
      double jdelz = x[j][2] - x[i][2];
      double rij_sq = jdelx*jdelx + jdely*jdely + jdelz*jdelz;

      if(rij_sq < cutoff*cutoff) {
        double rij = sqrt(rij_sq);
        const int jtype = atom->type[j];

        // Record bond information for future use
        nextTwoBodyInfo->tag = j;
        // TODO: I don't think that jj is the same as ii
        // nextTwoBodyInfo->globalTag = atom->tag[jj]-1;
        nextTwoBodyInfo->globalTag = atom->tag[j]-1;
        nextTwoBodyInfo->r = rij;
        nextTwoBodyInfo->del[0] = jdelx / rij;
        nextTwoBodyInfo->del[1] = jdely / rij;
        nextTwoBodyInfo->del[2] = jdelz / rij;
        nextTwoBodyInfo->f.resize(num_ffg);
        nextTwoBodyInfo->fprime.resize(num_ffg);

        // Loop over all FFG SVs
        for (int ffg_i=0; ffg_i<num_ffg; ffg_i++) {
          SVNode * ffg = ffg_nodes[itype-1][ffg_i];

          double partial_sum = 0;

          // Evaluate f_j
          // if false, uses f_k
          bool useFj = !(jtype == ffg->neighbor_types[0]);

          // TODO: going to be false when j/k are swapped

          nextTwoBodyInfo->f[ffg_i] = ffg->splines[useFj]->eval(rij, nextTwoBodyInfo->fprime[ffg_i]);
          
          for(int kk = 0; kk < numBonds; kk++) {
            const MEAM2Body& bondk = twoBodyInfo[kk];

            // Identify which type of FFG SV this triplet corresponds to
            int bondType = PairSplineTree::helper_pairing(
              jtype, atom->type[bondk.tag]
            );

            if (bondType != ffg->bondType) continue;

            double cos_theta = (nextTwoBodyInfo->del[0]*bondk.del[0] +
                                nextTwoBodyInfo->del[1]*bondk.del[1] +
                                nextTwoBodyInfo->del[2]*bondk.del[2]);

            // Evaluate f_k and g
            // Note: the last entry in the FFG splines is always the G spline
            partial_sum += bondk.f[ffg_i] * ffg->g->eval(cos_theta);
          }

          // Accumulate FFG contribution to energy
          double tmp = nextTwoBodyInfo->f[ffg_i]*partial_sum;
          per_atom_energy += tmp;
          ffg->energies[iGlobalTag] += tmp;  // used for tracking intermediate energies
          // ffg->energies[i] += tmp;  // used for tracking intermediate energies
        }

        // Loop over Rho SVs
        for (SVNode * rho : rho_nodes[itype-1]) {

          // Check if correct bond type
          if (jtype != rho->bondType) continue;

          double tmp = rho->splines[0]->eval(rij);
          per_atom_energy += tmp;
          rho->energies[iGlobalTag] += tmp;
          // rho->energies[i] += tmp;
        }

        numBonds++;
        nextTwoBodyInfo++;
      }
    }

    // Compute three-body contributions to force, if FFG SVs are used
    for (int ffg_i=0; ffg_i<num_ffg; ffg_i++) {
      double forces_i[3] = {0, 0, 0};

      SVNode * ffg = ffg_nodes[itype-1][ffg_i];

      for(int jj = 0; jj < numBonds; jj++) {
        const MEAM2Body bondj = twoBodyInfo[jj];
        double rij = bondj.r;
        int j = bondj.tag;

        double f_rij_prime = bondj.fprime[ffg_i];
        double f_rij = bondj.f[ffg_i];

        double forces_j[3] = {0, 0, 0};
        const int jtype = atom->type[j];

        MEAM2Body const* bondk = twoBodyInfo;
        for(int kk = 0; kk < jj; kk++, ++bondk) {

          int bondType = PairSplineTree::helper_pairing(
            jtype, atom->type[bondk->tag]
          );

          if (bondType != ffg->bondType) continue;

          double rik = bondk->r;

          double cos_theta = (bondj.del[0]*bondk->del[0] +
                              bondj.del[1]*bondk->del[1] +
                              bondj.del[2]*bondk->del[2]);
          double g_prime;
          // TODO: am I evaluating the G splines twice as much as necessary?
          double g_value = ffg->g->eval(cos_theta, g_prime);
          double f_rik_prime = bondk->fprime[ffg_i];
          double f_rik = bondk->f[ffg_i];

          double fij = -g_value * f_rik * f_rij_prime;
          double fik = -g_value * f_rij * f_rik_prime;

          double prefactor = f_rij * f_rik * g_prime;
          double prefactor_ij = prefactor / rij;
          double prefactor_ik = prefactor / rik;
          fij += prefactor_ij * cos_theta;
          fik += prefactor_ik * cos_theta;

          double fj[3], fk[3];

          fj[0] = bondj.del[0] * fij - bondk->del[0] * prefactor_ij;
          fj[1] = bondj.del[1] * fij - bondk->del[1] * prefactor_ij;
          fj[2] = bondj.del[2] * fij - bondk->del[2] * prefactor_ij;
          forces_j[0] += fj[0];
          forces_j[1] += fj[1];
          forces_j[2] += fj[2];

          // TODO: I think you're missing cross terms here; forces on j due to k?

          fk[0] = bondk->del[0] * fik - bondj.del[0] * prefactor_ik;
          fk[1] = bondk->del[1] * fik - bondj.del[1] * prefactor_ik;
          fk[2] = bondk->del[2] * fik - bondj.del[2] * prefactor_ik;
          forces_i[0] -= fk[0];
          forces_i[1] -= fk[1];
          forces_i[2] -= fk[2];

          int k = bondk->tag;

          // forces[k][0] += fk[0];
          // forces[k][1] += fk[1];
          // forces[k][2] += fk[2];

          // Forces on atom k due to atom i
          ffg->forces[iGlobalTag][bondk->globalTag][0] += fk[0];
          ffg->forces[iGlobalTag][bondk->globalTag][1] += fk[1];
          ffg->forces[iGlobalTag][bondk->globalTag][2] += fk[2];

          if(evflag) {
            double delta_ij[3];
            double delta_ik[3];
            delta_ij[0] = bondj.del[0] * rij;
            delta_ij[1] = bondj.del[1] * rij;
            delta_ij[2] = bondj.del[2] * rij;
            delta_ik[0] = bondk->del[0] * rik;
            delta_ik[1] = bondk->del[1] * rik;
            delta_ik[2] = bondk->del[2] * rik;
            ev_tally3(i, j, k, 0.0, 0.0, fj, fk, delta_ij, delta_ik);
          }
        }

        // forces[i][0] -= forces_j[0];
        // forces[i][1] -= forces_j[1];
        // forces[i][2] -= forces_j[2];
        // forces[j][0] += forces_j[0];
        // forces[j][1] += forces_j[1];
        // forces[j][2] += forces_j[2];

        // Forces on atom i due to atom j
        ffg->forces[iGlobalTag][iGlobalTag][0] -= forces_j[0];
        ffg->forces[iGlobalTag][iGlobalTag][1] -= forces_j[1];
        ffg->forces[iGlobalTag][iGlobalTag][2] -= forces_j[2];
        // Forces on atom j due to atom i
        ffg->forces[iGlobalTag][bondj.globalTag][0] += forces_j[0];
        ffg->forces[iGlobalTag][bondj.globalTag][1] += forces_j[1];
        ffg->forces[iGlobalTag][bondj.globalTag][2] += forces_j[2];
      }

      // Forces on atom i due to atoms k
      ffg->forces[iGlobalTag][iGlobalTag][0] += forces_i[0];
      ffg->forces[iGlobalTag][iGlobalTag][1] += forces_i[1];
      ffg->forces[iGlobalTag][iGlobalTag][2] += forces_i[2];
    }

    // forces[i][0] += forces_i[0];
    // forces[i][1] += forces_i[1];
    // forces[i][2] += forces_i[2];
  }

  comm->forward_comm_pair(this);

  // Compute two-body pair interactions
  for(int ii = 0; ii < listfull->inum; ii++) {
    int i = listfull->ilist[ii];
    const int itype = atom->type[i];

    for (SVNode * rho : rho_nodes[itype-1]) {
      for(int jj = 0; jj < listfull->numneigh[i]; jj++) {
        int j = listfull->firstneigh[i][jj];
        j &= NEIGHMASK;

        double jdel[3];
        jdel[0] = x[j][0] - x[i][0];
        jdel[1] = x[j][1] - x[i][1];
        jdel[2] = x[j][2] - x[i][2];
        double rij_sq = jdel[0]*jdel[0] + jdel[1]*jdel[1] + jdel[2]*jdel[2];

        if(rij_sq < cutoff*cutoff) {
          double rij = sqrt(rij_sq);
          const int jtype = atom->type[j];

          if (jtype != rho->bondType) continue;

          double rho_prime_j;
          rho->splines[0]->eval(rij, rho_prime_j);
          double fpair = rho_prime_j;
          double pair_pot = 0.0;

          // Divide by r_ij to get forces from gradient

          fpair /= rij;

          // forces[i][0] += jdel[0]*fpair;
          // forces[i][1] += jdel[1]*fpair;
          // forces[i][2] += jdel[2]*fpair;
          // forces[j][0] -= jdel[0]*fpair;
          // forces[j][1] -= jdel[1]*fpair;
          // forces[j][2] -= jdel[2]*fpair;

          // Forces on atom i due to atom j
          rho->forces[atom->tag[i]-1][atom->tag[i]-1][0] += jdel[0]*fpair;
          rho->forces[atom->tag[i]-1][atom->tag[i]-1][1] += jdel[1]*fpair;
          rho->forces[atom->tag[i]-1][atom->tag[i]-1][2] += jdel[2]*fpair;

          // Forces on atom j due to atom i
          rho->forces[atom->tag[i]-1][atom->tag[j]-1][0] -= jdel[0]*fpair;
          rho->forces[atom->tag[i]-1][atom->tag[j]-1][1] -= jdel[1]*fpair;
          rho->forces[atom->tag[i]-1][atom->tag[j]-1][2] -= jdel[2]*fpair;

          // TODO: may need to figure out how to update forces using local tags instead

          if (evflag) ev_tally(i, j, atom->nlocal, force->newton_pair,
                              pair_pot, 0.0, -fpair, jdel[0], jdel[1], jdel[2]);
        }
      }
    }
  }

  for (int itype=0; itype<nodes.size(); itype++) {

    // Constructs a list-of-lists where each sub-list is a sub-tree for a
    // function at a given recursion depth. The first node of a sub-tree
    // should always be a function. If only one node in tree, it must
    // be an SVNode.

    dvec  intermediateEng;
    dvec3 intermediateFcs;

    if (nodes[itype].size() == 1) {
      intermediateEng = nodes[itype][0]->energies;
      intermediateFcs = nodes[itype][0]->forces;

      if(eflag) {
        for(int ii = 0; ii < listfull->inum; ii++) {
          int i = atom->tag[ii]-1;
          // Accumulate per-atom energies
          if(eflag_global)
            eng_vdwl += intermediateEng[ii];
          if(eflag_atom)
            eatom[i] += intermediateEng[ii];

          // Accumulate per-atom forces
          for(int jj = 0; jj < listfull->inum; jj++) {
            int j = atom->tag[jj]-1;
            forces[j][0] += intermediateFcs[ii][jj][0];
            forces[j][1] += intermediateFcs[ii][jj][1];
            forces[j][2] += intermediateFcs[ii][jj][2];
          }
        }
      }
      continue;
    }
    else {
      std::vector<std::vector<Node *>> subTrees;

      for (Node * node : nodes[itype]) {
        bool isFxn = false;
        std::string nn = node->description;

        if ( arity1Functions.find(nn) != arity1Functions.end() )
          isFxn = true;
        else if ( arity2Functions.find(nn) != arity2Functions.end() )
          isFxn = true;

        if (isFxn) {
          // Start a new sub-tree
          subTrees.push_back({node});
        } else {
          // Grow the current deepest sub-tree
          subTrees.back().push_back(node);
        }

        // If the sub-tree is complete, evaluate its function
        std::vector<Node *> back = subTrees.back();
        int stackSize = back.size();
        while (stackSize == subTrees.back()[0]->arity + 1) {
          back = subTrees.back();

          std::vector<std::pair<dvec, dvec3>> args;

          // Prepare lists of arguments to be passed to function
          for (int i=1; i<back.size(); i++) {
            args.push_back(std::make_pair(back[i]->energies, back[i]->forces));
          }

          // Now evaluate stuff
          if (back[0]->arity == 1) {
            intermediateEng = back[0]->eval(args[0]);
            intermediateFcs = back[0]->deriv(args[0]);
          }
          else if (back[0]->arity == 2) {
            intermediateEng = back[0]->eval(args[0], args[1]);
            intermediateFcs = back[0]->deriv(args[0], args[1]);
          }
          
          if (subTrees.size() != 1) {  // Still some left to evaluate
            subTrees.pop_back();

            // Append intermediate results to sub-tree
            Node * dummyNode = new Node;
            dummyNode->energies = intermediateEng;
            dummyNode->forces   = intermediateFcs;
            dummyNode->description = "dummy";
            subTrees.back().push_back(dummyNode);
            stackSize = subTrees.back().size();
          } else {  // Done evaluating all sub-trees

            for(int ii = 0; ii < listfull->inum; ii++) {
              int i = atom->tag[ii]-1;

              // Accumulate per-atom energies
              if (eflag) {
                if(eflag_global)
                  eng_vdwl += intermediateEng[i];
                if(eflag_atom)
                  eatom[i] += intermediateEng[i];
              }

              // Accumulate per-atom forces
              for(int jj = 0; jj < listfull->inum; jj++) {
                int j = atom->tag[jj]-1;

                forces[j][0] += intermediateFcs[ii][jj][0];
                forces[j][1] += intermediateFcs[ii][jj][1];
                forces[j][2] += intermediateFcs[ii][jj][2];
              }
            }
            stackSize = -1;  // break condition
          }
        }
      }
    }
  }

  if(vflag_fdotr)
    virial_fdotr_compute();

}

/* ---------------------------------------------------------------------- */

void PairSplineTree::allocate()
{
  allocated = 1;
  int n = nelements;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  int nmultichoose2 = n*(n+1)/2;
  //Change the functional form
  //f_ij->f_i
  //g_i(cos\theta_ijk)->g_jk(cos\theta_ijk)

  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSplineTree::settings(int narg, char **/*arg*/)
{
  if(narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSplineTree::coeff(int narg, char **arg)
{
  int i,j,n;

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // ensure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read potential file: also sets the number of elements.
  read_file(arg, error);

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = sorted (alphabetically) list of element names

  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }

    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i], elements[j]) == 0)
        break;

    if (j < nelements) {
      map[i-2] = j;
    }
    else error->all(FLERR,"No matching element in potential file");
  }

  for (SVNode * svn : svnodes) {
    // Identify atom types of each neighbor
    int n = svn->neighbor_elements.size();
    svn->neighbor_types.resize(n);

    // Record host atom type
    int host_i;
    for (host_i=0; host_i<nelements; host_i++)
      if (strcmp(svn->hostElement.c_str(), elements[host_i]) == 0)
        svn->hostType = host_i;

    // Record neighbor types
    for (int ii=0; ii<svn->neighbor_elements.size(); ++ii) {
      std::string el = svn->neighbor_elements[ii];

      int jj;
      for (jj=0; jj<nelements; jj++)
        if (strcmp(el.c_str(), elements[jj]) == 0)
          svn->neighbor_types[ii] = jj+1;
    }

    bool isRho = svn->svType.compare("Rho") == 0;
    bool isFFG = svn->svType.compare("FFG") == 0;

    // Convert types to single integer for comparison later
    int bt;
    if (isRho)
      bt = PairSplineTree::helper_pairing(svn->neighbor_types[0]);
    else if (isFFG)
      if (n == 1)
        bt = PairSplineTree::helper_pairing(svn->neighbor_types[0], svn->neighbor_types[0]);
      else if (n == 2)
        bt = PairSplineTree::helper_pairing(svn->neighbor_types[0], svn->neighbor_types[1]);
    else
      error->all(FLERR, "Only 2- and 3-body SVs are supported");

    svn->bondType = bt;

    if (isRho) rho_nodes[svn->hostType].push_back(svn);
    else if (isFFG) ffg_nodes[svn->hostType].push_back(svn);
  }

  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

#define MAXLINE 1024

void PairSplineTree::read_file(char **arg, Error * error)
{
  const char* filename = arg[2];
  int nmultichoose2; // = (n+1)*n/2;

  if(comm->me == 0) {
    FILE *fp = force->open_potential(filename);
    if(fp == NULL) {
      char str[1024];
      snprintf(str,128,"Cannot open spline MEAM potential file %s", filename);
      error->one(FLERR,str);
    }

    // Skip first line of file. It's a comment.
    char line[MAXLINE];
    char *ptr;

    std::string line_str;

    while (fgets(line, MAXLINE, fp) != NULL) {
      line_str = std::string(line);

      if (line[0] != '#' && line[0] != '\n') {
        // Look for SV declaration first
        std::string three = line_str.substr(0, 3);
        int isRho = three.compare("Rho") == 0;
        int isFFG = three.compare("FFG") == 0;

        if (isRho || isFFG) {
          StructureVector sv = StructureVector();

          sv.parse(fp, error);
          sv.svType = three;

          sv_templates.emplace(sv.name, sv);
        }

        if (line_str.substr(0,4).compare("Tree") == 0) {

          utils::sfgets(FLERR, line, MAXLINE, fp, filename, error);
          std::string tree_str = line;
          tree_str = tree_str.substr(0, tree_str.size()-1);  // trailing newline

          std::vector<std::string> element_trees;
          element_trees = PairSplineTree::helper_split(tree_str, '|');

          for (int ii=0; ii<element_trees.size(); ++ii) {
            std::string s = element_trees[ii];

            // Remove all formatting
            s.erase(std::remove(s.begin(), s.end(), '<'), s.end());
            s.erase(std::remove(s.begin(), s.end(), '>'), s.end());
            std::replace(s.begin(), s.end(), '(', ' ');
            std::replace(s.begin(), s.end(), ')', ' ');
            std::replace(s.begin(), s.end(), ',', ' ');

            // Reduce all whitespace to a single space
            s = PairSplineTree::helper_reduce(s);

            element_trees[ii] = s;
          }

          for (std::string subtree: element_trees) {

            std::vector<std::string> tree_nodes;
            tree_nodes = PairSplineTree::helper_split(subtree, ' ');
            
            std::string el = tree_nodes[0];

            // Note: nelements gets incremented at the end of the loop
            nodes[nelements] = std::vector<Node *>();

            for (std::string nn: tree_nodes) {
              bool isFxn;
              bool isArity1 = false;

              if ( arity1Functions.find(nn) != arity1Functions.end() ) {
                isFxn = true;
                isArity1 = true;
              }
              else if ( arity2Functions.find(nn) != arity2Functions.end() ) {
                isFxn = true;
              }
              else {
                isFxn = false;
              }

              if (nn.compare(el) == 0) {
                // First "node" is the element symbol
                continue;
              } else if (isFxn) {
                // Add a function node
                Node * n = new Node;
                n->description = nn;

                if (isArity1) {
                  n->f1 = arity1Functions[nn].first;
                  n->d1 = arity1Functions[nn].second;
                  n->arity = 1;
                } else {
                  n->f2 = arity2Functions[nn].first;
                  n->d2 = arity2Functions[nn].second;
                  n->arity = 2;
                }

                nodes[nelements].push_back(n);
              } else {
                // Try to add a StructureVectorNode
                auto entry = sv_templates.find(nn);

                if (entry == sv_templates.end()) {
                  printf("Unable to find structure '%s' (%d, %s)\n", nn.c_str(), nn.compare(el), el.c_str());
                } else {
                  StructureVector sv = entry->second;

                  SVNode * svnode = new SVNode;

                  svnode->svType = sv.svType;
                  svnode->hostElement = el;
                  svnode->description = sv.name;
                  svnode->name = sv.name;
                  svnode->neighbor_elements = sv.neighbor_elements;
                  svnode->cutoffs[0] = sv.cutoffs[0];
                  svnode->cutoffs[1] = sv.cutoffs[1];
                  // svnode->components = sv.components;
                  svnode->num_knots = sv.num_knots;

                  nodes[nelements].push_back(svnode);
                  svnodes.push_back(svnode);
                }
              }
            }

            nelements += 1;
          }

          for (SVNode * svn : svnodes) {
            for (int nk : svn->num_knots) {
              svn->numKnots += nk;
              svn->numParams += nk + 2;
            }

            // +2 because of d0 and dN
            totalNumParams += svn->numParams;
          }
        }
        else if (line_str.substr(0,10).compare("Parameters") == 0) {
          int svIdx = 0;
          int added = 0;

          for (int ii=0; ii<totalNumParams; ii++) {
            utils::sfgets(FLERR, line, MAXLINE, fp, filename, error);
            double k = std::stod(std::string(line));

            svnodes[svIdx]->parameters.push_back(k);
            added++;

            if (added == svnodes[svIdx]->numParams) {
              // TODO: parameter file should also read in end-derivatives
              svIdx++;
              added = 0;
            }
          }
        }
      }
    }

    elements = new char*[nelements];
    int el_i = 0;
    for (int i=3; i<nelements+3; i++) {
      elements[el_i] = new char[strlen(arg[i])+1];
      strcpy(elements[el_i], arg[i]);

      el_i++;
    }

    nmultichoose2 = ((nelements+1)*nelements)/2;

    // allocate!!
    allocate();

    fclose(fp);
  }

  // TODO: set up communicate() for SVNodes
  // either set up on 0 then communicate, or communicate then set up on all ranks

  for (SVNode * svn : svnodes) {
    svn->splineSetup(error);

    if (svn->svType.compare("FFG") == 0) {
      svn->g = svn->splines[svn->splines.size()-1];
    }
  }

  // Transfer spline functions from master processor to all other processors.
  MPI_Bcast(&nelements, 1, MPI_INT, 0, world);
  MPI_Bcast(&nmultichoose2, 1, MPI_INT, 0, world);
  // allocate!!
  if (comm->me != 0) {
    allocate();
    elements = new char*[nelements];
  }
  for (int i = 0; i < nelements; ++i) {
    int n;
    if (comm->me == 0)
      n = strlen(elements[i]);
    MPI_Bcast(&n, 1, MPI_INT, 0, world);
    if (comm->me != 0)
      elements[i] = new char[n+1];
    MPI_Bcast(elements[i], n+1, MPI_CHAR, 0, world);
  }

  // Determine maximum cutoff radius of all relevant spline functions.
  cutoff = 0.0;
  for (SVNode * svn : svnodes) {
    for (auto sp : svn->splines) {
      if (sp->cutoff() > cutoff) {
        cutoff = sp->cutoff();
      }
    }
  }

  // Set LAMMPS pair interaction flags.
  for(int i = 1; i <= atom->ntypes; i++) {
    for(int j = 1; j <= atom->ntypes; j++) {
      // setflag[i][j] = 1;
      cutsq[i][j] = cutoff;
    }
  }

}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */
void PairSplineTree::init_style()
{
  if(force->newton_pair == 0)
    error->all(FLERR,"Pair style meam/spline requires newton pair on");

  // Need both full and half neighbor list.
  int irequest_full = neighbor->request(this,instance_me);
  neighbor->requests[irequest_full]->id = 1;
  neighbor->requests[irequest_full]->half = 0;
  neighbor->requests[irequest_full]->full = 1;
  int irequest_half = neighbor->request(this,instance_me);
  neighbor->requests[irequest_half]->id = 2;
  // neighbor->requests[irequest_half]->half = 1;
  // neighbor->requests[irequest_half]->halffull = 1;
  // neighbor->requests[irequest_half]->halffulllist = irequest_full;
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   half or full
------------------------------------------------------------------------- */
void PairSplineTree::init_list(int id, NeighList *ptr)
{
  if(id == 1) listfull = ptr;
  else if(id == 2) listhalf = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
double PairSplineTree::init_one(int /*i*/, int /*j*/)
{
  return cutoff;
}

/* ---------------------------------------------------------------------- */

int PairSplineTree::pack_forward_comm(int n, int *list, double *buf,
                                      int /*pbc_flag*/, int * /*pbc*/)
{
  int* list_iter = list;
  int* list_iter_end = list + n;
  // while(list_iter != list_iter_end)
  //   *buf++ = Uprime_values[*list_iter++];
  return n;
}

/* ---------------------------------------------------------------------- */

void PairSplineTree::unpack_forward_comm(int n, int first, double *buf)
{
  // memcpy(&Uprime_values[first], buf, n * sizeof(buf[0]));
}

/* ---------------------------------------------------------------------- */

int PairSplineTree::pack_reverse_comm(int /*n*/, int /*first*/, double * /*buf*/)
{
  return 0;
}

/* ---------------------------------------------------------------------- */

void PairSplineTree::unpack_reverse_comm(int /*n*/, int * /*list*/, double * /*buf*/)
{
}

/* ----------------------------------------------------------------------
   Returns memory usage of local atom-based arrays
------------------------------------------------------------------------- */
double PairSplineTree::memory_usage()
{
  return nmax * sizeof(double);        // The Uprime_values array.
}


/// Calculates the second derivatives at the knots of the cubic spline.
void PairSplineTree::StructureVector::SplineFunction::prepareSpline(Error* error)
{
  xmin = X[0];
  xmax = X[N-1];

  isGridSpline = true;
  h = (xmax-xmin)/(N-1);
  hsq = h*h;

  double* u = new double[N];
  Y2[0] = -0.5;
  u[0] = (3.0/(X[1]-X[0])) * ((Y[1]-Y[0])/(X[1]-X[0]) - deriv0);
  for(int i = 1; i <= N-2; i++) {
    double sig = (X[i]-X[i-1]) / (X[i+1]-X[i-1]);
    double p = sig * Y2[i-1] + 2.0;
    Y2[i] = (sig - 1.0) / p;
    u[i] = (Y[i+1]-Y[i]) / (X[i+1]-X[i]) - (Y[i]-Y[i-1])/(X[i]-X[i-1]);
    u[i] = (6.0 * u[i]/(X[i+1]-X[i-1]) - sig*u[i-1])/p;

    if(fabs(h*i+xmin - X[i]) > 1e-8)
      isGridSpline = false;
  }

  double qn = 0.5;
  double un = (3.0/(X[N-1]-X[N-2])) * (derivN - (Y[N-1]-Y[N-2])/(X[N-1]-X[N-2]));
  Y2[N-1] = (un - qn*u[N-2]) / (qn * Y2[N-2] + 1.0);
  for(int k = N-2; k >= 0; k--) {
    Y2[k] = Y2[k] * Y2[k+1] + u[k];
  }

  delete[] u;

#if !SPLINE_MEAM_SUPPORT_NON_GRID_SPLINES
  if(!isGridSpline)
    error->one(FLERR,"Support for MEAM potentials with non-uniform cubic splines has not been enabled in the MEAM potential code. Set SPLINE_MEAM_SUPPORT_NON_GRID_SPLINES in pair_spline_meam.h to 1 to enable it");
#endif

  // Shift the spline to X=0 to speed up interpolation.
  for(int i = 0; i < N; i++) {
    Xs[i] = X[i] - xmin;
#if !SPLINE_MEAM_SUPPORT_NON_GRID_SPLINES
    if(i < N-1) Ydelta[i] = (Y[i+1]-Y[i])/h;
    Y2[i] /= h*6.0;
#endif
  }
  inv_h = (1/h);
  xmax_shifted = xmax - xmin;
}

/// Broadcasts the spline function parameters to all processors.
void PairSplineTree::StructureVector::SplineFunction::communicate(MPI_Comm& world, int me)
{
  MPI_Bcast(&N, 1, MPI_INT, 0, world);
  MPI_Bcast(&deriv0, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&derivN, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&xmin, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&xmax, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&xmax_shifted, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&isGridSpline, 1, MPI_INT, 0, world);
  MPI_Bcast(&h, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&hsq, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&inv_h, 1, MPI_DOUBLE, 0, world);
  if(me != 0) {
    X = new double[N];
    Xs = new double[N];
    Y = new double[N];
    Y2 = new double[N];
    Ydelta = new double[N];
  }
  MPI_Bcast(X, N, MPI_DOUBLE, 0, world);
  MPI_Bcast(Xs, N, MPI_DOUBLE, 0, world);
  MPI_Bcast(Y, N, MPI_DOUBLE, 0, world);
  MPI_Bcast(Y2, N, MPI_DOUBLE, 0, world);
  MPI_Bcast(Ydelta, N, MPI_DOUBLE, 0, world);
}

/// Writes a Gnuplot script that plots the spline function.
///
/// This function is for debugging only!
void PairSplineTree::StructureVector::SplineFunction::writeGnuplot(const char* filename,
                                                  const char* title) const
{
  FILE* fp = fopen(filename, "w");
  fprintf(fp, "#!/usr/bin/env gnuplot\n");
  if(title) fprintf(fp, "set title \"%s\"\n", title);
  double tmin = X[0] - (X[N-1] - X[0]) * 0.05;
  double tmax = X[N-1] + (X[N-1] - X[0]) * 0.05;
  double delta = (tmax - tmin) / (N*200);
  fprintf(fp, "set xrange [%f:%f]\n", tmin, tmax);
  fprintf(fp, "plot '-' with lines notitle, '-' with points notitle pt 3 lc 3\n");
  for(double x = tmin; x <= tmax+1e-8; x += delta) {
    double y = eval(x);
    fprintf(fp, "%f %f\n", x, y);
  }
  fprintf(fp, "e\n");
  for(int i = 0; i < N; i++) {
    fprintf(fp, "%f %f\n", X[i], Y[i]);
  }
  fprintf(fp, "e\n");
  fclose(fp);
}

void PairSplineTree::StructureVector::parse(FILE* fp, Error* error) {
  char line[MAXLINE];

  std::string tmp_str;
  int pos;
  std::string tmp;

  // "name: <name>"
  utils::sfgets(FLERR, line, MAXLINE, fp, NULL, error);
  tmp_str = std::string(line);
  pos = tmp_str.find("name: ");

  name = tmp_str.substr(pos+6);
  name.erase(std::remove(name.begin(), name.end(), '\n'), name.end());

  // "neighbors: <neigh_1> <neigh_2> ...""
  utils::sfgets(FLERR, line, MAXLINE, fp, NULL, error);
  tmp_str = std::string(line);
  pos = tmp_str.find("neighbors: ");

  tmp_str = tmp_str.substr(pos+11);
  neighbor_elements = PairSplineTree::helper_split(tmp_str, ' ');

  tmp = neighbor_elements[neighbor_elements.size()-1];
  tmp.erase(std::remove(tmp.begin(), tmp.end(), '\n'), tmp.end());
  neighbor_elements[neighbor_elements.size()-1] = tmp;

  // sort alphabetically neighbor_elements here so that AB == BA
  std::sort(neighbor_elements.begin(), neighbor_elements.end());

  // "cutoffs: <inner_cut> <outer_cut>""
  utils::sfgets(FLERR, line, MAXLINE, fp, NULL, error);
  tmp_str = std::string(line);
  pos = tmp_str.find("cutoffs: ");

  tmp_str = tmp_str.substr(pos+9);
  std::vector<std::string> cutoffs_str = PairSplineTree::helper_split(tmp_str, ' ');
  cutoffs[0] = std::stod(cutoffs_str[0]);
  cutoffs[1] = std::stod(cutoffs_str[1]);

  // // "components: <inner_cut> <outer_cut>""
  // utils::sfgets(FLERR, line, MAXLINE, fp, NULL, error);
  // tmp_str = std::string(line);
  // pos = tmp_str.find("components: ");

  // tmp_str = tmp_str.substr(pos+12);
  // components = PairSplineTree::helper_split(tmp_str, ' ');

  // tmp = components[components.size()-1];
  // tmp.erase(std::remove(tmp.begin(), tmp.end(), '\n'), tmp.end());
  // components[components.size()-1] = tmp;

  // "num_knots: <n_1> <n_2> ...""
  utils::sfgets(FLERR, line, MAXLINE, fp, NULL, error);
  tmp_str = std::string(line);
  pos = tmp_str.find("num_knots: ");

  tmp_str = tmp_str.substr(pos+11);
  std::vector<std::string> num_knots_str = PairSplineTree::helper_split(tmp_str, ' ');

  for (int i=0; i<num_knots_str.size(); ++i) {
    std::string nn = num_knots_str[i];

    if (i == num_knots_str.size() - 1) nn = nn.substr(0, nn.size()-1);

    num_knots.push_back(std::stoi(nn));
  }

  // // "restrictions: <k_1> <v_1> ... <k_i> <v_i>, ...""
  // utils::sfgets(FLERR, line, MAXLINE, fp, NULL, error);
  // tmp_str = std::string(line);
  // pos = tmp_str.find("restrictions: ");

  // tmp_str = tmp_str.substr(pos+14);
  // std::vector<std::string> restrictions = PairSplineTree::helper_split(tmp_str, ',');

  // for (int i=0; i < restrictions.size(); ++i) {
  //   // Now loop over the list of restrictions for a given component

  //   if (i > 0) restrictions[i] = restrictions[i].substr(1, restrictions[i].size());

  //   std::vector<std::string> per_comp_restrs= PairSplineTree::helper_split(restrictions[i],' ');
    
  //   for (int j=0; j < per_comp_restrs.size(); j=j+2) {
  //     std::string rk = per_comp_restrs[j];
  //     std::string rv = per_comp_restrs[j+1];

  //     restricted_knots.push_back(std::stoi(per_comp_restrs[j]));
  //     restricted_values.push_back(std::stod(per_comp_restrs[j+1]));
  //   }

  //   restrictions_per_component.push_back(per_comp_restrs.size()/2);
  // }
}

void PairSplineTree::StructureVector::display() {
  printf("Name: %s\n", name.c_str());

  printf("Neighbors: ");
  for (int i=0; i < neighbor_elements.size(); ++i) {
    printf("%s ", neighbor_elements[i].c_str());
  }
  printf("\n");

  printf("Cutoffs: %.2f %.2f\n", cutoffs[0], cutoffs[1]);

  printf("Number of knots: ");
  for (int i=0; i < num_knots.size(); ++i) {
    printf("%d ", num_knots[i]);
  }
  printf("\n");
}

void PairSplineTree::StructureVector::splineSetup(Error * error) {
  // Builds a SplineFunction for each SV component

  int num_components = num_knots.size();

  int shift = 0;
  for (int s_i=0; s_i<num_components; s_i++) {
    int nn = num_knots[s_i];

    double x[nn];

    if (num_components == 1 || s_i != num_components-1) { // rho-/f-splines use radial cutoffs
      double h = (cutoffs[1] - cutoffs[0])/(nn-1);

      x[0] = cutoffs[0];
      for (int ii=1; ii<nn; ii++) {
        x[ii] = x[ii-1] + h;
      }
      x[nn-1] = cutoffs[1]; // just to make sure it's exactly the cutoff
    } else {  // g-splines use [-1, 1] cutoffs
      double h = 2.0/(nn-1);

      x[0] = -1;
      for (int ii=1; ii<nn; ii++) {
        x[ii] = x[ii-1] + h;
      }
      x[nn-1] = 1;
    }

    SplineFunction * s = new SplineFunction;
    std::vector<double> s_params;
    s_params = std::vector<double>(parameters.begin()+shift, parameters.begin()+shift+nn+2);

    s->init(nn, x, s_params);
    s->prepareSpline(error);
    splines.push_back(s);

    shift += nn+2;
  }
}

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
------------------------------------------------------------------------- */