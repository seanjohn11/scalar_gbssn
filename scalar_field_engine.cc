/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2020 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
//#include <deal.II/lac/solver_idr.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/hdf5.h>

//#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/fe/mapping_q1.h>

// Originally neccessary include files that may not be neccessary anymore.
/*
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <deal.II/base/convergence_table.h>
#include <deal.II/distributed/solution_transfer.h>
*/

#include <fstream>
#include <iomanip>
#include <iostream>

// The following file includes the CellwiseInverseMassMatrix data structure
// that we will use for the mass matrix inversion, the only new include
// file for this tutorial program:
#include <deal.II/matrix_free/operators.h>




namespace Scalar_Evolution
{
  using namespace dealii;

  // Similarly to the other matrix-free tutorial programs, we collect all
  // parameters that control the execution of the program at the top of the
  // file. Here we specify problem dimension and element degree number.
  constexpr unsigned int dimension            = 1;
  constexpr unsigned int fe_degree            = 3;
  constexpr unsigned int n_q_points_1d        = fe_degree + 1;
  constexpr unsigned int alt_q_points         = fe_degree + 1;
  constexpr unsigned int n_global_refinements = 6;
  constexpr double outer_boundary_value       = 5.;
  constexpr double inner_boundary             = 0.0;
  constexpr double individual_cell_error_limit= 1e-5;
  constexpr unsigned int output_spacing       = 1;
  constexpr unsigned int initial_output_spacing = 100;
  constexpr double cfl_factor                 = 0.05;
  constexpr double seed_amp                   = /*1e-2*/0.;
  constexpr double seed_sigma                 = 1.;
  constexpr bool uniform_refinement           = false;
  constexpr double mass                       = 0.;
  constexpr double tau_val                    = 1.;
  constexpr double eta_val                    = 10.;
  constexpr double last_residual              = 1.;
  // DON'T FORGET ABOUT TIME STEP THAT IS DEFINED MUCH FURTHER BELOW

  using Number = double;


  LinearAlgebra::distributed::Vector<double> psi_solution;
  LinearAlgebra::distributed::Vector<double> new_psi_solution;
  LinearAlgebra::distributed::Vector<double> xi_solution;
  LinearAlgebra::distributed::Vector<double> new_xi_solution;

  LinearAlgebra::distributed::Vector<double> conformal_solution;
  LinearAlgebra::distributed::Vector<double> new_conformal_solution;
  LinearAlgebra::distributed::Vector<double> alpha_solution;
  LinearAlgebra::distributed::Vector<double> new_alpha_solution;
  LinearAlgebra::distributed::Vector<double> gamma_rr_solution;
  LinearAlgebra::distributed::Vector<double> new_gamma_rr_solution;
  LinearAlgebra::distributed::Vector<double> gamma_tt_solution;
  LinearAlgebra::distributed::Vector<double> new_gamma_tt_solution;
  LinearAlgebra::distributed::Vector<double> K_solution;
  LinearAlgebra::distributed::Vector<double> new_K_solution;
  LinearAlgebra::distributed::Vector<double> A_rr_solution;
  LinearAlgebra::distributed::Vector<double> new_A_rr_solution;
  LinearAlgebra::distributed::Vector<double> A_tt_solution;
  LinearAlgebra::distributed::Vector<double> new_A_tt_solution;
  LinearAlgebra::distributed::Vector<double> lambda_solution;
  LinearAlgebra::distributed::Vector<double> new_lambda_solution;
  LinearAlgebra::distributed::Vector<double> diff_conf_solution;
  LinearAlgebra::distributed::Vector<double> diff_alpha_solution;
  LinearAlgebra::distributed::Vector<double> diff_gamma_rr_solution;
  LinearAlgebra::distributed::Vector<double> diff_gamma_tt_solution;
  LinearAlgebra::distributed::Vector<double> diff_psi_solution;
  // List all the other solution vectors_
  LinearAlgebra::distributed::Vector<double> modal_solution;
  //LinearAlgebra::distributed::Vector<double> a_convergence_solution;

  Vector<double>                             constraint_violation;
}

// All the headers made by Sean Johnson that include all the operators
// and all the helper functions used throughout.
// The order in which the include files are listed does matter.
#include "helper_functions.h"
//#include "IC_operator.h"
#include "diff_IC_operator.h"
#include "diff_IC_solver.h"
#include "diff_Evolve_solver.h"
#include "IC_U_operator.h"
#include "IC_V_operator.h"
#include "psi_operator.h"
#include "gamma_rr_operator.h"
#include "gamma_tt_operator.h"
#include "xi_operator.h"
#include "alpha_operator.h"
#include "k_operator.h"
#include "a_rr_operator.h"
#include "a_tt_operator.h"
#include "conformal_operator.h"
#include "lambda_operator.h"
#include "OE_operator.h"
//#include "IC_solver.h"

namespace Scalar_Evolution
{
  using namespace dealii;

  // Next off are some details of the time integrator, namely a Courant number
  // that scales the time step size in terms of the formula, as well as a
  // selection of a few low-storage Runge--Kutta methods.
  // We specify the Courant number per stage of the Runge--Kutta
  // scheme, as this gives a more realistic expression of the numerical cost
  // for schemes of various numbers of stages.
  enum LowStorageRungeKuttaScheme
  {
    stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_7_order_4, /* Tselios, Simos, 2007 */
    stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
    FE,
  };
  constexpr LowStorageRungeKuttaScheme lsrk_scheme = stage_5_order_4;
  constexpr LowStorageRungeKuttaScheme lsrk_scheme_evolve = FE;



  // @sect3{Low-storage explicit Runge--Kutta time integrators}

  // The next few lines implement a few low-storage variants of Runge--Kutta
  // methods. These methods have specific Butcher tableaux with coefficients
  // $b_i$ and $a_i$ as shown in the introduction. As usual in Runge--Kutta
  // method, we can deduce time steps, $c_i = \sum_{j=1}^{i-2} b_i + a_{i-1}$
  // from those coefficients. The main advantage of this kind of scheme is the
  // fact that only two vectors are needed per stage, namely the accumulated
  // part of the solution $\mathbf{w}$ (that will hold the solution
  // $\mathbf{w}^{n+1}$ at the new time $t^{n+1}$ after the last stage), the
  // update vector $\mathbf{r}_i$ that gets evaluated during the stages, plus
  // one vector $\mathbf{k}_i$ to hold the evaluation of the operator. Such a
  // Runge--Kutta setup reduces the memory storage and memory access. As the
  // memory bandwidth is often the performance-limiting factor on modern
  // hardware when the evaluation of the differential operator is
  // well-optimized, performance can be improved over standard time
  // integrators. This is true also when taking into account that a
  // conventional Runge--Kutta scheme might allow for slightly larger time
  // steps as more free parameters allow for better stability properties.
  //
  // In this tutorial programs, we concentrate on a few variants of
  // low-storage schemes defined in the article by Kennedy, Carpenter, and
  // Lewis (2000), as well as one variant described by Tselios and Simos
  // (2007). There is a large series of other schemes available, which could
  // be addressed by additional sets of coefficients or slightly different
  // update formulas.
  //
  // We define a single class for the four integrators, distinguished by the
  // enum described above. To each scheme, we then fill the vectors for the
  // $b_i$ and $a_i$ to the given variables in the class.
  class LowStorageRungeKuttaIntegrator
  {
  public:
    LowStorageRungeKuttaIntegrator(const LowStorageRungeKuttaScheme scheme)
    {
      TimeStepping::runge_kutta_method lsrk;
      // First comes the three-stage scheme of order three by Kennedy et al.
      // (2000). While its stability region is significantly smaller than for
      // the other schemes, it only involves three stages, so it is very
      // competitive in terms of the work per stage.
      switch (scheme)
        {
          case stage_3_order_3:
            {
              lsrk = TimeStepping::LOW_STORAGE_RK_STAGE3_ORDER3;
              TimeStepping::LowStorageRungeKutta<
                LinearAlgebra::distributed::Vector<Number>>
                rk_integrator(lsrk);
              rk_integrator.get_coefficients(ai, bi, ci);
              break;
            }

            // The next scheme is a five-stage scheme of order four, again
            // defined in the paper by Kennedy et al. (2000).
          case stage_5_order_4:
            {
              lsrk = TimeStepping::LOW_STORAGE_RK_STAGE5_ORDER4;
              TimeStepping::LowStorageRungeKutta<
                LinearAlgebra::distributed::Vector<Number>>
                rk_integrator(lsrk);
              rk_integrator.get_coefficients(ai, bi, ci);
              break;
            }

            // The following scheme of seven stages and order four has been
            // explicitly derived for acoustics problems. It is a balance of
            // accuracy for imaginary eigenvalues among fourth order schemes,
            // combined with a large stability region. Since DG schemes are
            // dissipative among the highest frequencies, this does not
            // necessarily translate to the highest possible time step per
            // stage. In the context of the present tutorial program, the
            // numerical flux plays a crucial role in the dissipation and thus
            // also the maximal stable time step size. For the modified
            // Lax--Friedrichs flux, this scheme is similar to the
            // `stage_5_order_4` scheme in terms of step size per stage if only
            // stability is considered, but somewhat less efficient for the HLL
            // flux.
          case stage_7_order_4:
            {
              lsrk = TimeStepping::LOW_STORAGE_RK_STAGE7_ORDER4;
              TimeStepping::LowStorageRungeKutta<
                LinearAlgebra::distributed::Vector<Number>>
                rk_integrator(lsrk);
              rk_integrator.get_coefficients(ai, bi, ci);
              break;
            }

            // The last scheme included here is the nine-stage scheme of order
            // five from Kennedy et al. (2000). It is the most accurate among
            // the schemes used here, but the higher order of accuracy
            // sacrifices some stability, so the step length normalized per
            // stage is less than for the fourth order schemes.
          case stage_9_order_5:
            {
              lsrk = TimeStepping::LOW_STORAGE_RK_STAGE9_ORDER5;
              TimeStepping::LowStorageRungeKutta<
                LinearAlgebra::distributed::Vector<Number>>
                rk_integrator(lsrk);
              rk_integrator.get_coefficients(ai, bi, ci);
              break;
            }

          case FE:
          {
            lsrk = TimeStepping::FORWARD_EULER;
            TimeStepping::ExplicitRungeKutta<
              LinearAlgebra::distributed::Vector<Number>>
              rk_integrator(lsrk);
              ai.push_back(0);
              bi.push_back(1);
              //bi.push_back(0);
              ci.push_back(0);
            //rk_integrator.get_coefficients(ai, bi, ci);
            break;
          }

          default:
            AssertThrow(false, ExcNotImplemented());
        }

    }

    unsigned int n_stages() const
    {
      return bi.size();
    }

    // The main function of the time integrator is to go through the stages,
    // evaluate the operator, prepare the $\mathbf{r}_i$ vector for the next
    // evaluation, and update the solution vector $\mathbf{w}$. We hand off
    // the work to the `pde_operator` involved in order to be able to merge
    // the vector operations of the Runge--Kutta setup with the evaluation of
    // the differential operator for better performance, so all we do here is
    // to delegate the vectors and coefficients.
    //
    // We separately call the operator for the first stage because we need
    // slightly modified arguments there: We evaluate the solution from
    // the old solution $\mathbf{w}^n$ rather than a $\mathbf r_i$ vector, so
    // the first argument is `solution`. We here let the stage vector
    // $\mathbf{r}_i$ also hold the temporary result of the evaluation, as it
    // is not used otherwise. For all subsequent stages, we use the vector
    // `vec_ki` as the second vector argument to store the result of the
    // operator evaluation. Finally, when we are at the last stage, we must
    // skip the computation of the vector $\mathbf{r}_{s+1}$ as there is no
    // coefficient $a_s$ available (nor will it be used).
    template <typename VectorType, typename Operator_0,typename Operator_1, typename Operator_2, int dim>
    void perform_time_step(const Operator_0 &pde_operator_0,
                           const Operator_1 &pde_operator_1,
                           const double    current_time,
                           const double    time_step,
                           VectorType &solution_0,
                           VectorType &solution_1,
                           VectorType &vec_ri_0,
                           VectorType &vec_ri_1,
                           VectorType &    vec_ki,
                           Operator_2 &diff_op,
                           const unsigned int &nlevels,
                           DoFHandler<dim> &dof_handler_DG) const
    {
      pde_operator_0.perform_stage(current_time,
                                 bi[0] * time_step,
                                 ai[0] * time_step,
                                 solution_0,
                                 vec_ri_0,
                                 solution_0,
                                 vec_ri_0);
       pde_operator_1.perform_stage(current_time,
                                  bi[0] * time_step,
                                  ai[0] * time_step,
                                  solution_1,
                                  vec_ri_1,
                                  solution_1,
                                  vec_ri_1);

      conformal_solution = vec_ri_0;
      xi_solution = vec_ri_1;

      for (unsigned int stage = 1; stage < bi.size(); ++stage)
        {

          diff_op.assemble_rhs();
          diff_op.solve(nlevels,dof_handler_DG);

          const double c_i = ci[stage];
          pde_operator_0.perform_stage(current_time + c_i * time_step,
                                     bi[stage] * time_step,
                                     (stage == bi.size() - 1 ?
                                        0 :
                                        ai[stage] * time_step),
                                     vec_ri_0,
                                     vec_ki,
                                     solution_0,
                                     vec_ri_0);

           pde_operator_1.perform_stage(current_time + c_i * time_step,
                                      bi[stage] * time_step,
                                      (stage == bi.size() - 1 ?
                                         0 :
                                         ai[stage] * time_step),
                                      vec_ri_1,
                                      vec_ki,
                                      solution_1,
                                      vec_ri_1);




         conformal_solution = vec_ri_0;
         xi_solution = vec_ri_1;
       }
       conformal_solution = solution_0;
       xi_solution = solution_1;

       //std::cout << "First Ci is: " << ci[1] << std::endl;
}

template <typename VectorType, typename Operator_0,typename Operator_1, typename Operator_2, typename Operator_3, typename Operator_4,
           typename Operator_5, typename Operator_6, typename Operator_7, typename Operator_8, typename Operator_9, typename Operator_10,int dim, typename Operator_11>
void perform_time_step_evolve(const Operator_0 &pde_operator_0,
                       const Operator_1 &pde_operator_1,
                       const Operator_2 &pde_operator_2,
                       const Operator_3 &pde_operator_3,
                       const Operator_4 &pde_operator_4,
                       const Operator_5 &pde_operator_5,
                       const Operator_6 &pde_operator_6,
                       const Operator_7 &pde_operator_7,
                       const Operator_8 &pde_operator_8,
                       const Operator_9 &pde_operator_9,
                       const double    current_time,
                       const double    time_step,
                       std::vector<VectorType> &    solution,
                       std::vector<VectorType> &    vec_ri,
                       VectorType &    vec_ki,
                       Operator_10 &diff_op,
                       const unsigned int &nlevels,
                       DoFHandler<dim> &dof_handler_DG,
                       const Mapping<dim> &mapping,
                       const FE_DGP<dim>  &fe_mod,
                       const Operator_11 &OE_operator,
                       const DoFHandler<dim> &dof_modal) const
{
  solution[0] = conformal_solution;
  solution[1] = gamma_rr_solution;
  solution[2] = gamma_tt_solution;
  solution[3] = K_solution;
  solution[4] = A_rr_solution;
  solution[5] = A_tt_solution;
  solution[6] = lambda_solution;
  solution[7] = psi_solution;
  solution[8] = xi_solution;
  solution[9] = alpha_solution;
  double u_avg;


  pde_operator_0.perform_stage(current_time,
                             bi[0] * time_step,
                             ai[0] * time_step,
                             solution[0],
                             vec_ri[0],
                             solution[0],
                             vec_ri[0]);

                             std::cout << "conformal" << std::endl;
                             u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[0], 0);
                             OE_operator.convert_modal(dof_handler_DG, vec_ri[0], dof_modal);
                             OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                             OE_operator.convert_nodal(dof_handler_DG, vec_ri[0], dof_modal);

   pde_operator_1.perform_stage(current_time,
                              bi[0] * time_step,
                              ai[0] * time_step,
                              solution[1],
                              vec_ri[1],
                              solution[1],
                              vec_ri[1]);

                              std::cout << "gamma_rr" << std::endl;
                              u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[1], 0);
                              OE_operator.convert_modal(dof_handler_DG, vec_ri[1], dof_modal);
                              OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                              OE_operator.convert_nodal(dof_handler_DG, vec_ri[1], dof_modal);

    pde_operator_2.perform_stage(current_time,
                               bi[0] * time_step,
                               ai[0] * time_step,
                               solution[2],
                               vec_ri[2],
                               solution[2],
                               vec_ri[2]);

                               std::cout << "gamma_tt" << std::endl;
                               u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[2], 0);
                               OE_operator.convert_modal(dof_handler_DG, vec_ri[2], dof_modal);
                               OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                               OE_operator.convert_nodal(dof_handler_DG, vec_ri[2], dof_modal);

     pde_operator_3.perform_stage(current_time,
                                bi[0] * time_step,
                                ai[0] * time_step,
                                solution[3],
                                vec_ri[3],
                                solution[3],
                                vec_ri[3]);

                                std::cout << "K" << std::endl;
                                u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[3], 0);
                                OE_operator.convert_modal(dof_handler_DG, vec_ri[3], dof_modal);
                                OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                OE_operator.convert_nodal(dof_handler_DG, vec_ri[3], dof_modal);

    pde_operator_4.perform_stage(current_time,
                               bi[0] * time_step,
                               ai[0] * time_step,
                               solution[4],
                               vec_ri[4],
                               solution[4],
                               vec_ri[4]);

                               std::cout << "A_rr" << std::endl;
                               u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[4], 0);
                               OE_operator.convert_modal(dof_handler_DG, vec_ri[4], dof_modal);
                               OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                               OE_operator.convert_nodal(dof_handler_DG, vec_ri[4], dof_modal);

     pde_operator_5.perform_stage(current_time,
                                bi[0] * time_step,
                                ai[0] * time_step,
                                solution[5],
                                vec_ri[5],
                                solution[5],
                                vec_ri[5]);

                                std::cout << "A_tt" << std::endl;
                                u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[5], 0);
                                OE_operator.convert_modal(dof_handler_DG, vec_ri[5], dof_modal);
                                OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                OE_operator.convert_nodal(dof_handler_DG, vec_ri[5], dof_modal);

  pde_operator_6.perform_stage(current_time,
                             bi[0] * time_step,
                             ai[0] * time_step,
                             solution[6],
                             vec_ri[6],
                             solution[6],
                             vec_ri[6]);

                             std::cout << "lambda" << std::endl;
                             u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[6], 0);
                             OE_operator.convert_modal(dof_handler_DG, vec_ri[6], dof_modal);
                             OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,false);
                             OE_operator.convert_nodal(dof_handler_DG, vec_ri[6], dof_modal);

   pde_operator_7.perform_stage(current_time,
                              bi[0] * time_step,
                              ai[0] * time_step,
                              solution[7],
                              vec_ri[7],
                              solution[7],
                              vec_ri[7]);

                              std::cout << "psi" << std::endl;
                              u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[7], 0);
                              OE_operator.convert_modal(dof_handler_DG, vec_ri[7], dof_modal);
                              OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                              OE_operator.convert_nodal(dof_handler_DG, vec_ri[7], dof_modal);

    pde_operator_8.perform_stage(current_time,
                               bi[0] * time_step,
                               ai[0] * time_step,
                               solution[8],
                               vec_ri[8],
                               solution[8],
                               vec_ri[8]);

                               std::cout << "xi" << std::endl;
                               u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[8], 0);
                               OE_operator.convert_modal(dof_handler_DG, vec_ri[8], dof_modal);
                               OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                               OE_operator.convert_nodal(dof_handler_DG, vec_ri[8], dof_modal);

   pde_operator_9.perform_stage(current_time,
                              bi[0] * time_step,
                              ai[0] * time_step,
                              solution[9],
                              vec_ri[9],
                              solution[9],
                              vec_ri[9]);

                              std::cout << "alpha" << std::endl;
                              u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[9], 0);
                              OE_operator.convert_modal(dof_handler_DG, vec_ri[9], dof_modal);
                              OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                              OE_operator.convert_nodal(dof_handler_DG, vec_ri[9], dof_modal);

  conformal_solution = vec_ri[0];
  gamma_rr_solution = vec_ri[1];
  gamma_tt_solution = vec_ri[2];
  K_solution = vec_ri[3];
  A_rr_solution = vec_ri[4];
  A_tt_solution = vec_ri[5];
  lambda_solution = vec_ri[6];
  psi_solution = vec_ri[7];
  xi_solution = vec_ri[8];
  alpha_solution = vec_ri[9];

  for (unsigned int stage = 1; stage < bi.size(); ++stage)
    {

      diff_op.assemble_rhs();
      diff_op.solve(nlevels,dof_handler_DG);

      const double c_i = ci[stage];
      pde_operator_0.perform_stage(current_time + c_i * time_step,
                                 bi[stage] * time_step,
                                 (stage == bi.size() - 1 ?
                                    0 :
                                    ai[stage] * time_step),
                                 vec_ri[0],
                                 vec_ki,
                                 solution[0],
                                 vec_ri[0]);

                                 u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[0], 0);
                                 OE_operator.convert_modal(dof_handler_DG, vec_ri[0], dof_modal);
                                 OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                 OE_operator.convert_nodal(dof_handler_DG, vec_ri[0], dof_modal);

       pde_operator_1.perform_stage(current_time + c_i * time_step,
                                  bi[stage] * time_step,
                                  (stage == bi.size() - 1 ?
                                     0 :
                                     ai[stage] * time_step),
                                  vec_ri[1],
                                  vec_ki,
                                  solution[1],
                                  vec_ri[1]);

                                  u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[1], 0);
                                  OE_operator.convert_modal(dof_handler_DG, vec_ri[1], dof_modal);
                                  OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                  OE_operator.convert_nodal(dof_handler_DG, vec_ri[1], dof_modal);

      pde_operator_2.perform_stage(current_time + c_i * time_step,
                                 bi[stage] * time_step,
                                 (stage == bi.size() - 1 ?
                                    0 :
                                    ai[stage] * time_step),
                                 vec_ri[2],
                                 vec_ki,
                                 solution[2],
                                 vec_ri[2]);

                                 u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[2], 0);
                                 OE_operator.convert_modal(dof_handler_DG, vec_ri[2], dof_modal);
                                 OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                 OE_operator.convert_nodal(dof_handler_DG, vec_ri[2], dof_modal);

     pde_operator_3.perform_stage(current_time + c_i * time_step,
                                bi[stage] * time_step,
                                (stage == bi.size() - 1 ?
                                   0 :
                                   ai[stage] * time_step),
                                vec_ri[3],
                                vec_ki,
                                solution[3],
                                vec_ri[3]);

                                u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[3], 0);
                                OE_operator.convert_modal(dof_handler_DG, vec_ri[3], dof_modal);
                                OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                OE_operator.convert_nodal(dof_handler_DG, vec_ri[3], dof_modal);

      pde_operator_4.perform_stage(current_time + c_i * time_step,
                                 bi[stage] * time_step,
                                 (stage == bi.size() - 1 ?
                                    0 :
                                    ai[stage] * time_step),
                                 vec_ri[4],
                                 vec_ki,
                                 solution[4],
                                 vec_ri[4]);

                                 u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[4], 0);
                                 OE_operator.convert_modal(dof_handler_DG, vec_ri[4], dof_modal);
                                 OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                 OE_operator.convert_nodal(dof_handler_DG, vec_ri[4], dof_modal);

       pde_operator_5.perform_stage(current_time + c_i * time_step,
                                  bi[stage] * time_step,
                                  (stage == bi.size() - 1 ?
                                     0 :
                                     ai[stage] * time_step),
                                  vec_ri[5],
                                  vec_ki,
                                  solution[5],
                                  vec_ri[5]);

                                  u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[5], 0);
                                  OE_operator.convert_modal(dof_handler_DG, vec_ri[5], dof_modal);
                                  OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                  OE_operator.convert_nodal(dof_handler_DG, vec_ri[5], dof_modal);

        pde_operator_6.perform_stage(current_time + c_i * time_step,
                                   bi[stage] * time_step,
                                   (stage == bi.size() - 1 ?
                                      0 :
                                      ai[stage] * time_step),
                                   vec_ri[6],
                                   vec_ki,
                                   solution[6],
                                   vec_ri[6]);

                                   u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[6], 0);
                                   OE_operator.convert_modal(dof_handler_DG, vec_ri[6], dof_modal);
                                   OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,false);
                                   OE_operator.convert_nodal(dof_handler_DG, vec_ri[6], dof_modal);

         pde_operator_7.perform_stage(current_time + c_i * time_step,
                                    bi[stage] * time_step,
                                    (stage == bi.size() - 1 ?
                                       0 :
                                       ai[stage] * time_step),
                                    vec_ri[7],
                                    vec_ki,
                                    solution[7],
                                    vec_ri[7]);

                                    u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[7], 0);
                                    OE_operator.convert_modal(dof_handler_DG, vec_ri[7], dof_modal);
                                    OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                    OE_operator.convert_nodal(dof_handler_DG, vec_ri[7], dof_modal);

        pde_operator_8.perform_stage(current_time + c_i * time_step,
                                   bi[stage] * time_step,
                                   (stage == bi.size() - 1 ?
                                      0 :
                                      ai[stage] * time_step),
                                   vec_ri[8],
                                   vec_ki,
                                   solution[8],
                                   vec_ri[8]);

                                   u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[8], 0);
                                   OE_operator.convert_modal(dof_handler_DG, vec_ri[8], dof_modal);
                                   OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                   OE_operator.convert_nodal(dof_handler_DG, vec_ri[8], dof_modal);

         pde_operator_9.perform_stage(current_time + c_i * time_step,
                                    bi[stage] * time_step,
                                    (stage == bi.size() - 1 ?
                                       0 :
                                       ai[stage] * time_step),
                                    vec_ri[9],
                                    vec_ki,
                                    solution[9],
                                    vec_ri[9]);

                                    u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), vec_ri[9], 0);
                                    OE_operator.convert_modal(dof_handler_DG, vec_ri[9], dof_modal);
                                    OE_operator.apply(time_step,u_avg,dof_modal,mapping,fe_mod,modal_solution,modal_solution,true);
                                    OE_operator.convert_nodal(dof_handler_DG, vec_ri[9], dof_modal);




    conformal_solution = vec_ri[0];
    gamma_rr_solution = vec_ri[1];
    gamma_tt_solution = vec_ri[2];
    K_solution = vec_ri[3];
    A_rr_solution = vec_ri[4];
    A_tt_solution = vec_ri[5];
    lambda_solution = vec_ri[6];
    psi_solution = vec_ri[7];
    xi_solution = vec_ri[8];
    alpha_solution = vec_ri[9];
   }
   conformal_solution = solution[0];
   gamma_rr_solution = solution[1];
   gamma_tt_solution = solution[2];
   K_solution = solution[3];
   A_rr_solution = solution[4];
   A_tt_solution = solution[5];
   lambda_solution = solution[6];
   psi_solution = solution[7];
   xi_solution = solution[8];
   alpha_solution = solution[9];

   //std::cout << "First Ci is: " << ci[1] << std::endl;
}
  private:
    std::vector<double> bi;
    std::vector<double> ai;
    std::vector<double> ci;
  };


  // This is the constructor for moving all the evolution along
  // The idea is to solve the initial conditions, then solve the evolution eqns
  // using a multistage approach where on each intermediary step or stage
  // the constraint equations are solved.
  template <int dim>
  class EvolutionProblem
  {
  public:
    EvolutionProblem();

    void run();

  private:

    void make_grid();

    void make_constraints();

    void make_dofs();

    void output_results_IC(const unsigned int result_number);

    void output_results_Evolve(const unsigned int result_number);

    void output_results_Constrain(const unsigned int result_number);

    void IC_gen();

    void CG_to_DG();

    void Hamiltonian_Violation();

    ConditionalOStream pcout;

  /*#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
  #else*/
    Triangulation<dim> triangulation;
  //#endif

    FESystem<dim>   fe_DG;
    FE_DGP<dim> fe_modal;
    MappingQ<dim>   mapping;
    DoFHandler<dim> dof_handler_DG;
    //DoFHandler<dim> dof_handler_CG;
    DoFHandler<dim> dof_handler_modal;

    AffineConstraints<double>            constraints_DG;
    AffineConstraints<double>            constraints_DG_odd;

    TimerOutput timer;

    /*Psi_Operator<dim, fe_degree, n_q_points_1d> psi_operator;
    //List all the other operators
    Pi_Operator<dim, fe_degree, n_q_points_1d> pi_operator;
    Phi_Operator<dim, fe_degree, n_q_points_1d> phi_operator;
    Alpha_Solver<dim> alpha_solver;*/
    //IC_Solver<dim> IC_solver;
    Diff_IC_Solver<dim> diff_IC_solver;
    Diff_Evolve_Solver<dim> diff_evolve_solver;
    VOperator<dim, fe_degree, n_q_points_1d> v_operator;
    UOperator<dim, fe_degree, n_q_points_1d> u_operator;
    Conformal_Operator<dim, fe_degree, n_q_points_1d> conf_operator;
    Gamma_rr_Operator<dim, fe_degree, n_q_points_1d> gamma_rr_operator;
    Gamma_tt_Operator<dim, fe_degree, n_q_points_1d> gamma_tt_operator;
    K_Operator<dim, fe_degree, n_q_points_1d> k_operator;
    A_rr_Operator<dim, fe_degree, n_q_points_1d> a_rr_operator;
    A_tt_Operator<dim, fe_degree, n_q_points_1d> a_tt_operator;
    Lambda_Operator<dim, fe_degree, n_q_points_1d> lambda_operator;
    Psi_Operator<dim, fe_degree, n_q_points_1d> psi_operator;
    Xi_Operator<dim, fe_degree, n_q_points_1d> xi_operator;
    Alpha_Operator<dim, fe_degree, n_q_points_1d> alpha_operator;
    OE_Operator<dim, fe_degree, n_q_points_1d> oe_operator;

    double time, time_step;
  };

  // The constructor for this class is unsurprising: We set up a parallel
  // triangulation based on the `MPI_COMM_WORLD` communicator, a vector finite
  // element with `dim+2` components for density, momentum, and energy, a
  // high-order mapping of the same degree as the underlying finite element,
  // and initialize the time and time step to zero.
  template <int dim>
  EvolutionProblem<dim>::EvolutionProblem()
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  /*#ifdef DEAL_II_WITH_P4EST
    , triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                      dim>::construct_multigrid_hierarchy)
  #else*/
    , triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
  //#endif
    , fe_DG(FE_DGQ<dim>(fe_degree))
    , fe_modal(fe_degree)
    , mapping(fe_degree)
    , dof_handler_DG(triangulation)
    , dof_handler_modal(triangulation)
    , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    /*, psi_operator(timer)
    , pi_operator(timer)
    , phi_operator(timer)
    , alpha_solver()*/
    , diff_IC_solver()
    , diff_evolve_solver()
    , v_operator(timer)
    , u_operator(timer)
    , conf_operator(timer)
    , gamma_rr_operator(timer)
    , gamma_tt_operator(timer)
    , k_operator(timer)
    , a_rr_operator(timer)
    , a_tt_operator(timer)
    , lambda_operator(timer)
    , psi_operator(timer)
    , xi_operator(timer)
    , alpha_operator(timer)
    , oe_operator(timer)
    , time(0)
    , time_step(0)
  {}

  // This makes the mesh and sets boundary id's for outer edges
  // On axis should have the id=2 and the outer edge parallel to the z-axis
  // should have an id=3 and then the outer edge parallel to the rho axis should
  // have an id=3.
  template <int dim>
  void EvolutionProblem<dim>::make_grid()
  {
    GridGenerator::hyper_cube(triangulation,
                                   inner_boundary,
                                   outer_boundary_value);


    triangulation.refine_global(n_global_refinements);

    for (const auto &cell : triangulation.cell_iterators())
    {
      for (const auto &face : cell->face_iterators())
        {
          const auto center = face->center();

          if (std::fabs(center(0) - (outer_boundary_value)) < 1e-12)
            face->set_boundary_id(1);
          else if  (std::fabs(center(0)) < 1e-12)
            face->set_boundary_id(0);
        }
      }
    }

    template <int dim>
    void EvolutionProblem<dim>::make_constraints()
    {
      //dof_handler_DG.distribute_dofs(fe_DG);

      /*const IndexSet locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler_CG);

      constraints.clear();
      constraints.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler_CG, constraints);
      constraints.close();

      constraints_a.clear();
      constraints_a.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler_CG, constraints_a);
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler_CG, 0, Functions::ConstantFunction<dim>(1.), constraints_a);
      constraints_a.close();

      constraints_alpha.clear();
      constraints_alpha.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler_CG, constraints_alpha);
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler_CG, 1,constraints_alpha(dof_handler_CG) , constraints_alpha);
      constraints_alpha.close();*/

      const IndexSet locally_relevant_dofs_DG =
        DoFTools::extract_locally_relevant_dofs(dof_handler_DG);

      constraints_DG.clear();
      constraints_DG.reinit(locally_relevant_dofs_DG);
      //DoFTools::make_hanging_node_constraints(dof_handler_DG, constraints_DG);
      constraints_DG.close();

      constraints_DG_odd.clear();
      constraints_DG_odd.reinit(locally_relevant_dofs_DG);
      //DoFTools::make_hanging_node_constraints(dof_handler_DG, constraints_DG_odd);


/*
          std::vector<unsigned int> local_dof_indices(fe_DG.dofs_per_cell);
          for (const auto &cell: dof_handler_DG.active_cell_iterators())
              for (const auto f : cell->face_indices())
                if (cell->at_boundary(f))
                  {
                    bool face_is_on_z_axis = false;
                    const auto center = cell->face(f)->center();

                    if (std::fabs(center(0))< 1e-12){
                      face_is_on_z_axis = true;
                      //std::cout << center(1) << std::endl;
                      //std::cout << center(0) << std::endl;
                    }

                    if (face_is_on_z_axis)
                     {
                       //std::cout << "This is an edge" << std::endl;
                       //std::cout << cell->face(f)->n_active_fe_indices() << std::endl;
                       //cell->face(f)->get_dof_indices(local_face_dof_indices);
                       cell->get_dof_indices(local_dof_indices);
                       std::vector<Point<dim>> supports;
                       supports = fe_DG.get_unit_support_points();
                       //std::cout << supports.size() << std::endl;
                       //std::cout << fe_DG.get_unit_support_points()[0][0] << std::endl;
                       for (unsigned int i=0; i<supports.size(); ++i){
                         Point<dim> dof_point;
                         dof_point = mapping.transform_unit_to_real_cell(cell,supports[i]);
                         if (std::fabs(dof_point(0)) < 1e-12){
                           //std::cout << local_dof_indices[i]<< "   " << dof_point[0] << "   " << dof_point[1] << std::endl;
                           //std::cout << dof_point[1] << std::endl;
                           //std::cout << dof_point[0] << std::endl;
                           constraints_DG_odd.add_line(local_dof_indices[i]);
                           //std::cout << "Good news?" << std::endl;
                         }
                         }
                       }
                     }
      //std::cout << constraints_DG_odd.n_constraints() << std::endl;
/*
      std::map<unsigned int, Point<2>> dof_location_map =
      DoFTools::map_dofs_to_support_points(mapping, dof_handler_DG);

      std::ofstream dof_location_file("dof-locations.gnuplot");
      DoFTools::write_gnuplot_dof_support_point_info(dof_location_file,
                                                   dof_location_map);
*/


      constraints_DG_odd.close();
    }

    template <int dim>
    void EvolutionProblem<dim>::make_dofs()
    {
    // This still needs some editting to make up for all the additional
    // variables that are being solved for.
    //const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler_DG, &dof_handler_DG/*, &dof_handler_CG, &dof_handler_CG, &dof_handler_CG*/};
    //const std::vector<const AffineConstraints<double> *> constraints_list = {&constraints_DG, &constraints_DG_odd/*, &constraints, &constraints_a, &constraints_alpha*/};

    conf_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    gamma_rr_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    gamma_rr_operator.initialize_vector(gamma_rr_solution);
    gamma_rr_operator.initialize_vector(new_gamma_rr_solution);
    VectorTools::interpolate(mapping,dof_handler_DG,Functions::ConstantFunction<dim>(1.),gamma_rr_solution);
    gamma_tt_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    gamma_tt_operator.initialize_vector(gamma_tt_solution);
    gamma_tt_operator.initialize_vector(new_gamma_tt_solution);
    VectorTools::interpolate(mapping,dof_handler_DG,Functions::ConstantFunction<dim>(1.),gamma_tt_solution);
    k_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    k_operator.initialize_vector(K_solution);
    k_operator.initialize_vector(new_K_solution);
    VectorTools::interpolate(mapping,dof_handler_DG,Functions::ZeroFunction<dim>(),K_solution);
    a_rr_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    a_rr_operator.initialize_vector(A_rr_solution);
    a_rr_operator.initialize_vector(new_A_rr_solution);
    VectorTools::interpolate(mapping,dof_handler_DG,Functions::ZeroFunction<dim>(),A_rr_solution);
    a_tt_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    a_tt_operator.initialize_vector(A_tt_solution);
    a_tt_operator.initialize_vector(new_A_tt_solution);
    VectorTools::interpolate(mapping,dof_handler_DG,Functions::ZeroFunction<dim>(),A_tt_solution);
    lambda_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    lambda_operator.initialize_vector(lambda_solution);
    lambda_operator.initialize_vector(new_lambda_solution);
    VectorTools::interpolate(mapping,dof_handler_DG,Functions::ZeroFunction<dim>(),lambda_solution);
    psi_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    psi_operator.initialize_vector(psi_solution);
    psi_operator.initialize_vector(new_psi_solution);
    VectorTools::interpolate(mapping,dof_handler_DG,phi_init<dim>(),psi_solution);
    xi_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    xi_operator.initialize_vector(xi_solution);
    xi_operator.initialize_vector(new_xi_solution);
    VectorTools::interpolate(mapping,dof_handler_DG,Functions::ZeroFunction<dim>(),xi_solution);
    alpha_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    alpha_operator.initialize_vector(alpha_solution);
    alpha_operator.initialize_vector(new_alpha_solution);
    VectorTools::interpolate(mapping,dof_handler_DG,Functions::ConstantFunction<dim>(1.),alpha_solution);


    dof_handler_modal.distribute_dofs(fe_modal);
    oe_operator.reinit(mapping,dof_handler_modal);
    oe_operator.initialize_vector(modal_solution);

    /*pi_operator.reinit(mapping,dof_handlers,constraints_list);
    pi_operator.initialize_vector(pi_solution);*/

    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "Number of degrees of freedom: " << dof_handler_DG.n_dofs()
          << " ( = " << " [vars] x "
          << triangulation.n_global_active_cells() << " [cells] x "
          << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
          << std::endl;
    pcout.get_stream().imbue(s);
  }




  //
  // For parallel programs, it is often instructive to look at the partitioning
  // of cells among processors. To this end, one can pass a vector of numbers
  // to DataOut::add_data_vector() that contains as many entries as the
  // current processor has active cells; these numbers should then be the
  // rank of the processor that owns each of these cells. Such a vector
  // could, for example, be obtained from
  // GridTools::get_subdomain_association(). On the other hand, on each MPI
  // process, DataOut will only read those entries that correspond to locally
  // owned cells, and these of course all have the same value: namely, the rank
  // of the current process. What is in the remaining entries of the vector
  // doesn't actually matter, and so we can just get away with a cheap trick: We
  // just fill *all* values of the vector we give to DataOut::add_data_vector()
  // with the rank of the current MPI process. The key is that on each process,
  // only the entries corresponding to the locally owned cells will be read,
  // ignoring the (wrong) values in other entries. The fact that every process
  // submits a vector in which the correct subset of entries is correct is all
  // that is necessary.
  template <int dim>
  void EvolutionProblem<dim>::output_results_IC(const unsigned int result_number)
  {

    /*pcout << "Time:" << std::setw(8) << std::setprecision(3) << time
          << ", dt: " << std::setw(8) << std::setprecision(2) << time_step
          << ", " << quantity_name << " L2: " << std::setprecision(4)
          << std::setw(10) << errors << std::endl;*/

    {
      TimerOutput::Scope t(timer, "output");

      //Postprocessor postprocessor;
      DataOut<dim>  data_out;
      /*
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);*/

      data_out.attach_dof_handler(dof_handler_DG);
      /*data_out.add_data_vector(psi_solution,"psi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(xi_solution, "xi",DataOut<dim>::type_dof_data);*/
      data_out.add_data_vector(conformal_solution, "conformal_phi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(xi_solution, "pi",DataOut<dim>::type_dof_data);
      /*data_out.add_data_vector(gamma_rr_solution, "gamma_rr",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(gamma_tt_solution, "gamma_tt",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(A_rr_solution, "A_rr",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(A_tt_solution, "A_tt",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(lambda_solution, "lambda",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(alpha_solution, "alpha",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(K_solution, "K",DataOut<dim>::type_dof_data);*/

      Vector<double> mpi_owner(triangulation.n_active_cells());
      mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      data_out.add_data_vector(mpi_owner, "owner",DataOut<dim>::type_cell_data);
      data_out.add_data_vector(constraint_violation, "violation", DataOut<dim>::type_cell_data);
      data_out.build_patches(mapping,
                             fe_DG.degree);

      const std::string filename =
        "Initial_" + Utilities::int_to_string(result_number, 3) + ".vtu";

      data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

    }
  }

  template <int dim>
  void EvolutionProblem<dim>::output_results_Evolve(const unsigned int result_number)
  {

    {
      TimerOutput::Scope t(timer, "output");

      //Postprocessor postprocessor;
      DataOut<dim>  data_out;

      /*DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);*/


      data_out.attach_dof_handler(dof_handler_DG);
      data_out.add_data_vector(psi_solution,"psi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(xi_solution, "xi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(conformal_solution, "conformal_phi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(gamma_rr_solution, "gamma_rr",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(gamma_tt_solution, "gamma_tt",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(A_rr_solution, "A_rr",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(A_tt_solution, "A_tt",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(lambda_solution, "lambda",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(alpha_solution, "alpha",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(K_solution, "K",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(diff_conf_solution, "dr_conf",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(diff_alpha_solution, "dr_alpha",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(diff_gamma_rr_solution, "dr_gamma_rr",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(diff_gamma_tt_solution, "dr_gamma_tt",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(diff_psi_solution, "dr_psi",DataOut<dim>::type_dof_data);

      Vector<double> mpi_owner(triangulation.n_active_cells());
      mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      data_out.add_data_vector(mpi_owner, "owner",DataOut<dim>::type_cell_data);
      data_out.add_data_vector(constraint_violation,"Constraint_Violation",DataOut<dim>::type_cell_data);
      //pcout << "Made it this far." <<std::endl;
      data_out.build_patches(mapping,
                             fe_DG.degree);

      const std::string filename =
        "Evolutions_" + Utilities::int_to_string(result_number, 3) + ".vtu";

      data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
    }
  }

  /*template <int dim>
  void EvolutionProblem<dim>::output_results_Constrain(const unsigned int result_number)
  {

    {
      TimerOutput::Scope t(timer, "output");

      //Postprocessor postprocessor;
      DataOut<dim>  data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.attach_dof_handler(dof_handler_CG);
      data_out.add_data_vector(alpha_solution,"alpha");
      data_out.add_data_vector(a_solution, "a");

      Vector<double> mpi_owner(triangulation.n_active_cells());
      mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      data_out.add_data_vector(mpi_owner, "owner");
      //pcout << "Made it this far." <<std::endl;
      data_out.build_patches(mapping,
                             fe_CG.degree);

      const std::string filename =
        "Constraints_" + Utilities::int_to_string(result_number, 3) + ".vtu";

      data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
    }
  }*/

  template <int dim>
  void EvolutionProblem<dim>::CG_to_DG()
  {
  }

  template <int dim>
  void EvolutionProblem<dim>::IC_gen()
  {
    const unsigned int nlevels = triangulation.n_global_levels();
    diff_IC_solver.setup_system(nlevels, mapping, dof_handler_DG, fe_DG);
    v_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    v_operator.initialize_vector(xi_solution);
    v_operator.initialize_vector(new_xi_solution);
    v_operator.initialize_vector(new_K_solution);
    v_operator.initialize_vector(new_lambda_solution);
    v_operator.initialize_vector(gamma_rr_solution);
    u_operator.reinit(mapping,dof_handler_DG,constraints_DG);
    u_operator.initialize_vector(conformal_solution);
    u_operator.initialize_vector(new_conformal_solution);
    u_operator.initialize_vector(new_psi_solution);

    //conformal_solution = 0.1;
    //pi_solution = 0.1;

    const LowStorageRungeKuttaIntegrator integrator(lsrk_scheme);

    double last_residual = 1.0;
    unsigned int cycle = 0;
    while (last_residual > individual_cell_error_limit && cycle < 1093*initial_output_spacing)
    {
      pcout << "Cycle" << cycle << std::endl;

      if (cycle > 0)
      {
        /*IC_refinement(IC_error_estimate);
        for (const auto &cell : triangulation.cell_iterators())
        {
          for (const auto &face : cell->face_iterators())
            {
              const auto center = face->center();

              if (std::fabs(center(0) - (outer_boundary)) < 1e-12)
                face->set_boundary_id(1);
              else if  (std::fabs(center(0) < 1e-12)
                face->set_boundary_id(0);
            }
        }*/
      }
      diff_IC_solver.assemble_rhs();
      diff_IC_solver.solve(nlevels,dof_handler_DG);
      new_conformal_solution = conformal_solution;
      new_xi_solution = xi_solution;

      double min_vertex_distance = std::numeric_limits<double>::max();
     for (const auto &cell : triangulation.active_cell_iterators())
       if (cell->is_locally_owned())
         min_vertex_distance =
           std::min(min_vertex_distance, cell->minimum_vertex_distance());
     min_vertex_distance =
       Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

       std::cout << "dx: " << min_vertex_distance << std::endl;

     time_step = cfl_factor*min_vertex_distance;
     std::cout << "Time step: " << time_step << std::endl;

      integrator.perform_time_step(u_operator,
                                        v_operator,
                                        time,
                                        time_step,
                                        new_conformal_solution,
                                        new_xi_solution,
                                        new_psi_solution,
                                        new_K_solution,
                                        new_lambda_solution,
                                        diff_IC_solver,
                                        triangulation.n_global_levels(),
                                        dof_handler_DG);
      //std::cout << pi_solution.l2_norm() << std::endl;
      //std::cout << conformal_solution.l2_norm() << std::endl;
      Hamiltonian_Violation();
      std::cout << constraint_violation.l2_norm() << std::endl;
      if (cycle % initial_output_spacing == 0){
      output_results_IC(cycle/initial_output_spacing);}
      cycle += 1;
      gamma_rr_solution = xi_solution;
      gamma_rr_solution.sadd(1.,-eta_val,conformal_solution);
      std::cout << gamma_rr_solution.l2_norm() << std::endl;
      last_residual = gamma_rr_solution.l2_norm();

      /*diff_IC_solver.setup_system(nlevels,mapping,dof_handler_DG);
      diff_IC_solver.assemble_rhs();
      diff_IC_solver.solve(nlevels,dof_handler_DG);
      IC_solver.assemble_rhs();
      IC_solver.solve(nlevels, dof_handler_DG);
      conformal_solution.sadd(1.,new_conformal_solution);
      output_results_Evolve(0);
      std::cout << new_conformal_solution.l2_norm() << std::endl;
      last_residual = new_conformal_solution.l2_norm();
      new_conformal_solution = 0.;*/

/*
      while(last_residual_norm > 1e-6){
        std::cout << "I made it to 1" << std::endl;
        a_solver.system_rhs = 0.;
        a_solver.a_update = 0.;
        a_solver.assemble_rhs();
        output_results_IC(cycle);
        std::cout << "I made it to 2" << std::endl;
        a_solver.solve(nlevels, dof_handler_DG);
        output_results_IC(cycle);
        std::cout << last_residual_norm << std::endl;
        cycle += 1;
      }
      output_results_IC(cycle);
      pcout << std::endl;*/

    }
  }

  /*

   template <int dim>
   void EvolutionProblem<dim>::IC_refinement(double &error_estimate)
   {
     Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

     const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler_CG);
     LinearAlgebra::distributed::Vector<double> evaluation_point(psi_solution);
     evaluation_point.reinit(dof_handler_CG.locally_owned_dofs(),
                     locally_relevant_dofs,
                     triangulation.get_communicator());
     evaluation_point.copy_locally_owned_data_from(psi_solution);
     constraints_empty.distribute(evaluation_point);
     evaluation_point.update_ghost_values();
     //evaluation_point = solution;

     const QGauss<dim> quadrature_formula(fe_degree + 2);

     FEValues<dim>     fe_values(fe_CG,
                             quadrature_formula,
                             update_values | update_hessians | update_gradients | update_quadrature_points |
                               update_JxW_values);



     const unsigned int dofs_per_cell = fe_CG.n_dofs_per_cell();
     const unsigned int n_q_points    = quadrature_formula.size();

     std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

     std::vector<double>  current_lap(n_q_points);
     std::vector<Tensor<1, dim>>  current_grad(n_q_points);
     std::vector<double>  current_values(n_q_points);
     int iterator_count=0;

     for (const auto &cell : dof_handler_CG.active_cell_iterators())
       {
         //estimated_error_per_cell = 0;
         fe_values.reinit(cell);


         fe_values.get_function_laplacians(evaluation_point, current_lap);
         fe_values.get_function_values(evaluation_point,current_values);
         fe_values.get_function_gradients(evaluation_point,current_grad);

         Coefficient<dim> coefficient_fun;


         for (unsigned int q = 0; q < n_q_points; ++q)
           {

             const double current_coefficient =
             coefficient_fun.value(fe_values.quadrature_point(q));

             estimated_error_per_cell(iterator_count) += ((current_lap[q] + current_grad[q][0]/fe_values.quadrature_point(q)(0)                // \lap u_n
                             + current_coefficient      // + tau/4
                             * (1 + current_values[q]))        // * psi_n
                             *(current_lap[q] + current_grad[q][0]/fe_values.quadrature_point(q)(0) + current_coefficient * (1 + current_values[q])))   // squared
                              fe_values.quadrature_point(q)(0)* * fe_values.JxW(q);             // rho * dx
           }
           estimated_error_per_cell(iterator_count) = sqrt(estimated_error_per_cell(iterator_count));
           //std::cout << estimated_error_per_cell(iterator_count);
           iterator_count += 1;

     }

     error_estimate = estimated_error_per_cell.linfty_norm();
     std::cout << error_estimate << std::endl;
     if (!uniform_refinement){
     GridRefinement::refine(triangulation,estimated_error_per_cell,individual_cell_error_limit/2);
     GridRefinement::coarsen(triangulation,estimated_error_per_cell,individual_cell_error_limit/400);

     triangulation.prepare_coarsening_and_refinement();


     triangulation.execute_coarsening_and_refinement();
   }
     else{
       if (error_estimate > individual_cell_error_limit){
       triangulation.refine_global(1);
     }
     }
   }*/

   template <int dim>
   void EvolutionProblem<dim>::Hamiltonian_Violation()
   {
     constraint_violation.reinit(0);
     constraint_violation.reinit(triangulation.n_active_cells());

     const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler_DG);
     LinearAlgebra::distributed::Vector<double> conf_point(conformal_solution);
     conf_point.reinit(dof_handler_DG.locally_owned_dofs(),
                     locally_relevant_dofs,
                     triangulation.get_communicator());
     conf_point.copy_locally_owned_data_from(conformal_solution);
     constraints_DG.distribute(conf_point);

     conf_point.update_ghost_values();
     //evaluation_point = solution;

     const QGauss<dim> quadrature_formula(fe_degree + 2);

     FEValues<dim>     fe_values(fe_DG,
                             quadrature_formula,
                             update_values | update_hessians | update_gradients | update_quadrature_points |
                               update_JxW_values);



     //const unsigned int dofs_per_cell = fe_DG.n_dofs_per_cell();
     const unsigned int n_q_points    = quadrature_formula.size();

     //std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

     std::vector<double>  current_lap(n_q_points);
     std::vector<Tensor<1, dim>>  current_grad(n_q_points);
     std::vector<double>  current_values(n_q_points);
     int iterator_count=0;

     pow_func<dim> pow_f;
     exp_func<dim> exp_f;
     H_func<dim> H_f;

     for (const auto &cell : dof_handler_DG.active_cell_iterators())
       {
         fe_values.reinit(cell);


         fe_values.get_function_laplacians(conf_point, current_lap);
         fe_values.get_function_values(conf_point,current_values);
         fe_values.get_function_gradients(conf_point,current_grad);

         for (unsigned int q = 0; q < n_q_points; ++q)
           {
             constraint_violation(iterator_count) += (exp_f.eval(4.*current_values[q])*(
                            current_lap[q]
                            + current_grad[q]*current_grad[q]
                            + 2/fe_values.quadrature_point(q)[0] * current_grad[q][0]
                          ) + H_f.value(fe_values.quadrature_point(q)) )        // * psi_n/4
                            *(exp_f.eval(4.*current_values[q])*(
                                           current_lap[q]
                                           + current_grad[q]*current_grad[q]
                                           + 2/fe_values.quadrature_point(q)[0] * current_grad[q][0]
                                         )+ H_f.value(fe_values.quadrature_point(q)))  // squared
                            * fe_values.quadrature_point(q)[0] * fe_values.quadrature_point(q)[0]* fe_values.JxW(q);             // * dx
           }
           constraint_violation(iterator_count) = sqrt(constraint_violation(iterator_count));
           iterator_count += 1;
   }
 }



   // The EvolutionProblem::run() function puts all pieces together. It starts off
   // by calling the function that creates the mesh and sets up data structures,
   // and then initializing the time integrator and the two temporary vectors of
   // the low-storage integrator. We call these vectors `rk_register_1` and
   // `rk_register_2`, and use the first vector to represent the quantity
   // $\mathbf{r}_i$ and the second one for $\mathbf{k}_i$ in the formulas for
   // the Runge--Kutta scheme outlined in the introduction. Before we start the
   // time loop, we compute the time step size by the
   // `WaveOperator::compute_cell_transport_speed()` function. For reasons of
   // comparison, we compare the result obtained there with the minimal mesh
   // size and print them to screen. For velocities and speeds of sound close
   // to unity as in this tutorial program, the predicted effective mesh size
   // will be close, but they could vary if scaling were different.
   template <int dim>
   void EvolutionProblem<dim>::run()
   {
     {
       const unsigned int n_vect_number = VectorizedArray<Number>::size();
       const unsigned int n_vect_bits   = 8 * sizeof(Number) * n_vect_number;

       pcout << "Running with "
             << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
             << " MPI processes" << std::endl;
       pcout << "Vectorization over " << n_vect_number << ' '
             << (std::is_same<Number, double>::value ? "doubles" : "floats")
             << " = " << n_vect_bits << " bits ("
             << Utilities::System::get_current_vectorization_level() << ')'
             << std::endl;
     }

     make_grid();

     make_constraints();

     IC_gen();

     make_dofs();




     double min_vertex_distance = std::numeric_limits<double>::max();
     for (const auto &cell : triangulation.active_cell_iterators())
       if (cell->is_locally_owned())
         min_vertex_distance =
           std::min(min_vertex_distance, cell->minimum_vertex_distance());
     min_vertex_distance =
       Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

     //double tot_time = 0;
     time_step = cfl_factor*min_vertex_distance;
     //tot_time += time_step;
     //time_step = 0.;
     pcout << "Time step size: " << time_step
           << ", minimal h: " << min_vertex_distance
           << std::endl
           << std::endl;



     LinearAlgebra::distributed::Vector<double> rk_register_1;
     LinearAlgebra::distributed::Vector<double> rk_register_2;
     LinearAlgebra::distributed::Vector<double> rk_register_3;
     LinearAlgebra::distributed::Vector<double> rk_register_4;
     LinearAlgebra::distributed::Vector<double> rk_register_5;
     LinearAlgebra::distributed::Vector<double> rk_register_6;
     LinearAlgebra::distributed::Vector<double> rk_register_7;
     LinearAlgebra::distributed::Vector<double> rk_register_8;
     LinearAlgebra::distributed::Vector<double> rk_register_9;
     LinearAlgebra::distributed::Vector<double> rk_register_10;
     LinearAlgebra::distributed::Vector<double> rk_register_11;
     rk_register_1.reinit(xi_solution);
     rk_register_2.reinit(xi_solution);
     rk_register_3.reinit(xi_solution);
     rk_register_4.reinit(xi_solution);
     rk_register_5.reinit(xi_solution);
     rk_register_6.reinit(xi_solution);
     rk_register_7.reinit(xi_solution);
     rk_register_8.reinit(xi_solution);
     rk_register_9.reinit(xi_solution);
     rk_register_10.reinit(xi_solution);
     rk_register_11.reinit(xi_solution);


     // Now we are ready to start the time loop, which we run until the time
     // has reached the desired end time. Every 5 time steps, we compute a new
     // estimate for the time step -- since the solution is nonlinear, it is
     // most effective to adapt the value during the course of the
     // simulation. In case the Courant number was chosen too aggressively, the
     // simulation will typically blow up with time step NaN, so that is easy
     // to detect here. One thing to note is that roundoff errors might
     // propagate to the leading digits due to an interaction of slightly
     // different time step selections that in turn lead to slightly different
     // solutions. To decrease this sensitivity, it is common practice to round
     // or truncate the time step size to a few digits, e.g. 3 in this case. In
     // case the current time is near the prescribed 'tick' value for output
     // (e.g. 0.02), we also write the output. After the end of the time loop,
     // we summarize the computation by printing some statistics, which is
     // mostly done by the TimerOutput::print_wall_time_statistics() function.
     unsigned int timestep_number = 0;

     std::vector<LinearAlgebra::distributed::Vector<double>> rk_register_first = {rk_register_1,rk_register_2,rk_register_3,rk_register_4,rk_register_5,rk_register_6,rk_register_7,rk_register_8,rk_register_9, rk_register_10};
     //std::vector<LinearAlgebra::distributed::Vector<double>> rk_register_second = {rk_register_2,rk_register_4,rk_register_6,rk_register_8,rk_register_10};
     std::vector<LinearAlgebra::distributed::Vector<double>> solutions_vec = {new_conformal_solution,
                                                                              new_gamma_rr_solution,
                                                                              new_gamma_tt_solution,
                                                                              new_K_solution,
                                                                              new_A_rr_solution,
                                                                              new_A_tt_solution,
                                                                              new_lambda_solution,
                                                                              new_psi_solution,
                                                                              new_xi_solution,
                                                                              new_alpha_solution
                                                                            };

     diff_evolve_solver.setup_system(triangulation.n_global_levels(),mapping,dof_handler_DG,fe_DG);
     diff_evolve_solver.assemble_rhs();
     diff_evolve_solver.solve(triangulation.n_global_levels(),dof_handler_DG);

     /*oe_operator.limiter(dof_handler_DG, mapping, fe_DG, diff_alpha_solution, diff_alpha_solution, 0.);
     oe_operator.limiter(dof_handler_DG, mapping, fe_DG, diff_gamma_rr_solution, diff_gamma_rr_solution, 0.);
     oe_operator.limiter(dof_handler_DG, mapping, fe_DG, diff_gamma_tt_solution, diff_gamma_tt_solution, 0.);*/
/*
     double u_avg;
     u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), diff_conf_solution, 0);
     oe_operator.convert_modal(dof_handler_DG, diff_conf_solution, dof_handler_modal);
     oe_operator.apply(time_step*10e8,u_avg,dof_handler_modal,mapping,fe_modal,modal_solution,modal_solution);
     oe_operator.convert_nodal(dof_handler_DG, diff_conf_solution, dof_handler_modal);

     u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), diff_alpha_solution, 0);
     oe_operator.convert_modal(dof_handler_DG, diff_alpha_solution, dof_handler_modal);
     oe_operator.apply(time_step*10e8,u_avg,dof_handler_modal,mapping,fe_modal,modal_solution,modal_solution);
     oe_operator.convert_nodal(dof_handler_DG, diff_alpha_solution, dof_handler_modal);

     u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), diff_gamma_rr_solution, 0);
     oe_operator.convert_modal(dof_handler_DG, diff_gamma_rr_solution, dof_handler_modal);
     oe_operator.apply(time_step*10e8,u_avg,dof_handler_modal,mapping,fe_modal,modal_solution,modal_solution);
     oe_operator.convert_nodal(dof_handler_DG, diff_gamma_rr_solution, dof_handler_modal);

     u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), diff_gamma_tt_solution, 0);
     oe_operator.convert_modal(dof_handler_DG, diff_gamma_tt_solution, dof_handler_modal);
     oe_operator.apply(time_step*10e8,u_avg,dof_handler_modal,mapping,fe_modal,modal_solution,modal_solution);
     oe_operator.convert_nodal(dof_handler_DG, diff_gamma_tt_solution, dof_handler_modal);*/



     output_results_Evolve(0);


     //dof_handler_modal.distribute_dofs(fe_modal);
     //dof_handler_modal.distribute_mg_dofs();

     const LowStorageRungeKuttaIntegrator integrator(lsrk_scheme_evolve);


     while (timestep_number < 100)//(time < final_time - 1e-12)
       {
         ++timestep_number;

         {
           new_conformal_solution = conformal_solution;
           new_gamma_rr_solution = gamma_rr_solution;
           new_gamma_tt_solution = gamma_tt_solution;
           new_K_solution = K_solution;
           new_A_rr_solution = A_rr_solution;
           new_A_tt_solution = A_tt_solution;
           new_lambda_solution = lambda_solution;
           new_psi_solution = psi_solution;
           new_xi_solution = xi_solution;
           new_alpha_solution = alpha_solution;

           TimerOutput::Scope t(timer, "rk time stepping total");
           integrator.perform_time_step_evolve(conf_operator,
                                        gamma_rr_operator,
                                        gamma_tt_operator,
                                        k_operator,
                                        a_rr_operator,
                                        a_tt_operator,
                                        lambda_operator,
                                        psi_operator,
                                        xi_operator,
                                        alpha_operator,
                                        time,
                                        time_step,
                                        solutions_vec,
                                        rk_register_first,
                                        rk_register_11,
                                        diff_evolve_solver,
                                        triangulation.n_global_levels(),
                                        dof_handler_DG,
                                        mapping,
                                        fe_modal,
                                        oe_operator,
                                        dof_handler_modal);
         }
         /*oe_operator.limiter(dof_handler_DG, mapping, fe_DG, K_solution, K_solution, true);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, conformal_solution, conformal_solution, true);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, gamma_rr_solution, gamma_rr_solution, true);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, gamma_tt_solution, gamma_tt_solution, true);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, A_rr_solution, A_rr_solution, true);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, A_tt_solution, A_tt_solution, true);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, psi_solution, psi_solution, true);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, xi_solution, xi_solution, true);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, lambda_solution, lambda_solution, false);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, alpha_solution, alpha_solution, true);*/



         Hamiltonian_Violation();
         time += time_step;

         std::cout << A_tt_solution.l2_norm() << std::endl;

         diff_evolve_solver.assemble_rhs();
         diff_evolve_solver.solve(triangulation.n_global_levels(),dof_handler_DG);

         /*u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), diff_conf_solution, 0);
         oe_operator.convert_modal(dof_handler_DG, diff_conf_solution, dof_handler_modal);
         oe_operator.apply(time_step*10e8,u_avg,dof_handler_modal,mapping,fe_modal,modal_solution,modal_solution);
         oe_operator.convert_nodal(dof_handler_DG, diff_conf_solution, dof_handler_modal);

         u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), diff_alpha_solution, 0);
         oe_operator.convert_modal(dof_handler_DG, diff_alpha_solution, dof_handler_modal);
         oe_operator.apply(time_step*10e8,u_avg,dof_handler_modal,mapping,fe_modal,modal_solution,modal_solution);
         oe_operator.convert_nodal(dof_handler_DG, diff_alpha_solution, dof_handler_modal);

         u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), diff_gamma_rr_solution, 0);
         oe_operator.convert_modal(dof_handler_DG, diff_gamma_rr_solution, dof_handler_modal);
         oe_operator.apply(time_step*10e8,u_avg,dof_handler_modal,mapping,fe_modal,modal_solution,modal_solution);
         oe_operator.convert_nodal(dof_handler_DG, diff_gamma_rr_solution, dof_handler_modal);

         u_avg = VectorTools::compute_mean_value(mapping, dof_handler_DG, QGauss<dim>(alt_q_points), diff_gamma_tt_solution, 0);
         oe_operator.convert_modal(dof_handler_DG, diff_gamma_tt_solution, dof_handler_modal);
         oe_operator.apply(time_step*10e8,u_avg,dof_handler_modal,mapping,fe_modal,modal_solution,modal_solution);
         oe_operator.convert_nodal(dof_handler_DG, diff_gamma_tt_solution, dof_handler_modal);*/

         /*oe_operator.limiter(dof_handler_DG, mapping, fe_DG, diff_alpha_solution, diff_alpha_solution, false);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, diff_gamma_rr_solution, diff_gamma_rr_solution, false);
         oe_operator.limiter(dof_handler_DG, mapping, fe_DG, diff_gamma_tt_solution, diff_gamma_tt_solution, false);*/

           if(timestep_number % output_spacing == 0){
             output_results_Evolve((timestep_number/output_spacing));
           }


     timer.print_wall_time_statistics(MPI_COMM_WORLD);
     pcout << std::endl;
     std::cout << time << std::endl;
     std::cout << timestep_number << std::endl;
   }
 }

// This is remanants of an old elliptic solver I was making that worked for
// all elliptic pdes but abandoned because need a solver that works
// both inside this EvolutionProblem class and outs*10e8ide during the intermediary
// time steps.
// It has been kept so I can steal already typed code from it.





} // namespace Brill_Evolution



  // The main() function is not surprising and follows what was done in all
  // previous MPI programs: As we run an MPI program, we need to call `MPI_Init()`
  // and `MPI_Finalize()`, which we do through the
  // Utilities::MPI::MPI_InitFinalize data structure. Note that we run the program
  // only with MPI, and set the thread count to 1.
  int main(int argc, char **argv)
  {
  using namespace Scalar_Evolution;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {
      deallog.depth_console(0);

      EvolutionProblem<dimension> evolution_problem;
      evolution_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
  }
