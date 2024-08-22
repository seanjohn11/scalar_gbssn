namespace Scalar_Evolution
{
  using namespace dealii;

  template <int dim>
    class IC_Solver
    {
    public:
      IC_Solver();
      void run();


      void setup_system(const unsigned int &nlevels,
                        const MappingQ<dim> &mapping,
                        DoFHandler<dim> &dof_handler_DG,
                        const FESystem<dim>   &fe_DG);
      void assemble_rhs();

      void solve(const unsigned int &nlevels, const DoFHandler<dim> &dof_handler_DG);

    private:
      AffineConstraints<double> constraints_DG;
      AffineConstraints<double> constraints_DG_odd;
      using SystemMatrixType =
          IC_Operator<dim, fe_degree, double>;
      SystemMatrixType IC_matrix;

      MGConstrainedDoFs mg_constrained_dofs_DG;
      MGConstrainedDoFs mg_constrained_dofs_DG_odd;
      using LevelMatrixType = IC_Operator<dim, fe_degree, float>;
      MGLevelObject<LevelMatrixType> IC_matrices;

      LinearAlgebra::distributed::Vector<double> IC_rhs;

      double             setup_time;
      ConditionalOStream pcout;
      ConditionalOStream time_details;

      void face_IC_apply(const MatrixFree<dim, double> &                   data,
                  LinearAlgebra::distributed::Vector<double> &      dst,
                  const LinearAlgebra::distributed::Vector<double> &src,
                  const std::pair<unsigned int, unsigned int> &face_range) const;

      void IC_cell_apply(const MatrixFree<dim, double> &                   data,
                  LinearAlgebra::distributed::Vector<double> &      dst,
                  const LinearAlgebra::distributed::Vector<double> &src,
                  const std::pair<unsigned int, unsigned int> &cell_range) const;

      void IC_boundary_apply(const MatrixFree<dim, double> &                   data,
                  LinearAlgebra::distributed::Vector<double> &      dst,
                  const LinearAlgebra::distributed::Vector<double> &src,
                  const std::pair<unsigned int, unsigned int> &face_range) const;
      pow_func<dim> pow_f;

    };



    // When we initialize the finite element, we of course have to use the
    // degree specified at the top of the file as well (otherwise, an exception
    // will be thrown at some point, since the computational kernel defined in
    // the templated LaplaceOperator class and the information from the finite
    // element read out by MatrixFree will not match). The constructor of the
    // triangulation needs to set an additional flag that tells the grid to
    // conform to the 2:1 cell balance over vertices, which is needed for the
    // convergence of the geometric multigrid routines. For the distributed
    // grid, we also need to specifically enable the multigrid hierarchy.
    template <int dim>
    IC_Solver<dim>::IC_Solver()
      : setup_time(0.)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      // The IC_Solver class holds an additional output stream that
      // collects detailed timings about the setup phase. This stream, called
      // time_details, is disabled by default through the @p false argument
      // specified here. For detailed timings, removing the @p false argument
      // prints all the details.
      , time_details(std::cout,
                     false &&
                       Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {}



    // @sect4{IC_Solver::setup_system}

    template <int dim>
    void IC_Solver<dim>::setup_system(const unsigned int &nlevels,
            const MappingQ<dim> &mapping,
            DoFHandler<dim> &dof_handler_DG,
            const FESystem<dim>   &fe_DG)
    {
      Timer time;
      setup_time = 0;

      IC_matrix.clear();
      IC_matrices.clear_elements();

      dof_handler_DG.distribute_dofs(fe_DG);
      dof_handler_DG.distribute_mg_dofs();

      const IndexSet locally_relevant_dofs_DG =
        DoFTools::extract_locally_relevant_dofs(dof_handler_DG);

      constraints_DG.clear();
      constraints_DG.reinit(locally_relevant_dofs_DG);
      constraints_DG.close();

      const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler_DG};
      const std::vector<const AffineConstraints<double> *> constraints_list = {&constraints_DG};

      setup_time += time.wall_time();
      time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time()
                   << "s/" << time.wall_time() << 's' << std::endl;
      time.restart();

      const std::vector<Quadrature<dim>> quadratures = {QGauss<dim>(fe_degree + 1),
                                                      QGauss<dim>(alt_q_points)};

      {
        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim, double>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_values | update_gradients | update_JxW_values | update_quadrature_points);
        additional_data.mapping_update_flags_inner_faces =
          (update_values | update_JxW_values | update_quadrature_points | update_normal_vectors);
        additional_data.mapping_update_flags_boundary_faces =
          (update_values | update_JxW_values | update_quadrature_points);
        std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(
          new MatrixFree<dim, double>());
        system_mf_storage->reinit(mapping,
                                  dof_handlers,
                                  constraints_list,
                                  quadratures,
                                  additional_data);
        IC_matrix.initialize(system_mf_storage,{0});

        IC_matrix.initialize_dof_vector(IC_rhs);
        IC_matrix.initialize_dof_vector(psi_solution);
        IC_matrix.initialize_dof_vector(xi_solution);
        IC_matrix.initialize_dof_vector(gamma_rr_solution);
        IC_matrix.initialize_dof_vector(gamma_tt_solution);
        IC_matrix.initialize_dof_vector(A_rr_solution);
        IC_matrix.initialize_dof_vector(A_tt_solution);
        IC_matrix.initialize_dof_vector(K_solution);
        IC_matrix.initialize_dof_vector(lambda_solution);
        IC_matrix.initialize_dof_vector(alpha_solution);
        IC_matrix.initialize_dof_vector(conformal_solution);
        IC_matrix.initialize_dof_vector(new_conformal_solution);

        IC_matrix.initialize_dof_vector(diff_conf_solution);
        IC_matrix.initialize_dof_vector(diff_new_conf_solution);

        VectorTools::interpolate(dof_handler_DG, phi_init<dim>(),psi_solution);
        xi_solution = 0.;
        gamma_rr_solution = 1.;
        gamma_tt_solution = 1.;
        alpha_solution = 1.;
        A_rr_solution = 0.;
        A_tt_solution = 0.;
        K_solution = 0.;
        lambda_solution = 0.;
        conformal_solution = 0.;

        IC_matrix.temp_conf = conformal_solution;
        IC_matrix.temp_diff_conf = diff_conf_solution;
        IC_matrix.temp_diff_new_conf = diff_new_conf_solution;


      }

      //system_matrix.initialize_dof_vector(beta_z_solution);
      //system_matrix.initialize_dof_vector(system_rhs);

      setup_time += time.wall_time();
      time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time()
                   << "s/" << time.wall_time() << 's' << std::endl;
      time.restart();

      // Next, initialize the matrices for the multigrid method on all the
      // levels. The data structure MGConstrainedDoFs keeps information about
      // the indices subject to boundary conditions as well as the indices on
      // edges between different refinement levels as described in the step-16
      // tutorial program. We then go through the levels of the mesh and
      // construct the constraints and matrices on each level. These follow
      // closely the construction of the system matrix on the original mesh,
      // except the slight difference in naming when accessing information on
      // the levels rather than the active cells.
      //const unsigned int nlevels = triangulation.n_global_levels();
      IC_matrices.resize(0, nlevels - 1);

      mg_constrained_dofs_DG.initialize(dof_handler_DG);


      MGLevelObject<LinearAlgebra::distributed::Vector<float>> conf_solution_mg;
      MGLevelObject<LinearAlgebra::distributed::Vector<float>> diff_conf_solution_mg;
      MGLevelObject<LinearAlgebra::distributed::Vector<float>> diff_new_conf_solution_mg;

      MGTransferMatrixFree<dim, float> mg_transfer_DG(mg_constrained_dofs_DG);
      mg_transfer_DG.build(dof_handler_DG);
      conf_solution_mg.resize(0, nlevels-1);
      diff_conf_solution_mg.resize(0, nlevels-1);
      diff_new_conf_solution_mg.resize(0, nlevels-1);
      mg_transfer_DG.interpolate_to_mg(dof_handler_DG,conf_solution_mg,conformal_solution);
      mg_transfer_DG.interpolate_to_mg(dof_handler_DG,diff_conf_solution_mg,diff_conf_solution);
      mg_transfer_DG.interpolate_to_mg(dof_handler_DG,diff_new_conf_solution_mg,diff_new_conf_solution);


      for (unsigned int level = 0; level < nlevels; ++level)
        {
          //const IndexSet relevant_dofs =
            //DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
          const IndexSet relevant_dofs_DG=
            DoFTools::extract_locally_relevant_level_dofs(dof_handler_DG, level);
          AffineConstraints<double> level_constraints_DG;

          level_constraints_DG.reinit(relevant_dofs_DG);
          level_constraints_DG.close();

          const std::vector<const AffineConstraints<double> *> level_constraints_list = {&level_constraints_DG};

          typename MatrixFree<dim, float>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme =
            MatrixFree<dim, float>::AdditionalData::none;
          additional_data.mapping_update_flags =
            (update_values | update_JxW_values | update_quadrature_points);
          additional_data.mapping_update_flags_boundary_faces =
            (update_values | update_JxW_values | update_quadrature_points);
          additional_data.mg_level = level;
          std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(
            new MatrixFree<dim, float>());
          mg_mf_storage_level->reinit(mapping,
                                      dof_handlers,
                                      level_constraints_list,
                                      quadratures,
                                      additional_data);

          IC_matrices[level].initialize(mg_mf_storage_level,
                                        mg_constrained_dofs_DG,
                                        level,
                                        {0});

          IC_matrices[level].temp_conf = conf_solution_mg[level];
          IC_matrices[level].temp_diff_conf = diff_conf_solution_mg[level];
          IC_matrices[level].temp_diff_new_conf = diff_new_conf_solution_mg[level];

        }
      setup_time += time.wall_time();
      time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time()
                   << "s/" << time.wall_time() << 's' << std::endl;
    }



    // @sect4{IC_Solver::assemble_rhs}

    // The assemble function is very simple since all we have to do is to
    // assemble the right hand side. Thanks to FEEvaluation and all the data
    // cached in the MatrixFree class, which we query from
    // MatrixFreeOperators::Base, this can be done in a few lines. Since this
    // call is not wrapped into a MatrixFree::cell_loop (which would be an
    // alternative), we must not forget to call compress() at the end of the
    // assembly to send all the contributions of the right hand side to the
    // owner of the respective degree of freedom.
    template <int dim>
    void IC_Solver<dim>::assemble_rhs()
    {

      this->IC_matrix.get_matrix_free()->cell_loop(&IC_Solver::IC_cell_apply,
                       /*&IC_Solver::face_IC_apply,
                       &IC_Solver::IC_boundary_apply,*/
                       this, IC_rhs, new_conformal_solution,
                       true
                     );
    }

    template <int dim>
    void IC_Solver<dim>::IC_cell_apply(const MatrixFree<dim, double> &                   data,
                LinearAlgebra::distributed::Vector<double> &      dst,
                const LinearAlgebra::distributed::Vector<double> &,
                const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, fe_degree, alt_q_points, 1, Number> new_conf(data,0,1);
      FEEvaluation<dim, fe_degree, alt_q_points, 1, Number> conf(data,0,1);
      FEEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff_conf(data,0,1);
      H_func <dim> H_f;
      exp_func<dim> exp_f;
      pow_func<dim> pow_f;


      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
           {
             new_conf.reinit(cell);

             conf.reinit(cell);
             conf.gather_evaluate(conformal_solution,true,true);
             diff_conf.reinit(cell);
             diff_conf.gather_evaluate(diff_conf_solution,true,true);


             for (unsigned int q=0; q < new_conf.n_q_points; ++q){
               VectorizedArray<double> r_val = new_conf.quadrature_point(q)[0];
               new_conf.submit_value(-exp_f.eval(4.*conf.get_value(q))*pow_f.eval(r_val,-1)*
               (diff_conf.get_value(q)
                + 2.*r_val*conf.get_gradient(q)[0]*diff_conf.get_value(q)
                + r_val*diff_conf.get_gradient(q)[0])
               -H_f.value(new_conf.quadrature_point(q))
               ,q);
             }
             new_conf.integrate(EvaluationFlags::values);
             new_conf.distribute_local_to_global(dst);

           }
      }

      template <int dim>
      void IC_Solver<dim>::face_IC_apply(const MatrixFree<dim, double> &data  ,
                  LinearAlgebra::distributed::Vector<double> &dst      ,
                  const LinearAlgebra::distributed::Vector<double> &,
                  const std::pair<unsigned int, unsigned int> &face_range) const
      {
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> new_conf_in(data,true,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> new_conf_out(data,false,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> conf_in(data,true,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> conf_out(data,false,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff_conf_in(data,true,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff_conf_out(data,false,0,1);

        exp_func<dim> exp_f;

        for (unsigned int face = face_range.first; face < face_range.second; ++face){
          new_conf_in.reinit(face);
          new_conf_out.reinit(face);
          conf_in.reinit(face);
          conf_in.gather_evaluate(conformal_solution, true, false);
          conf_out.reinit(face);
          conf_out.gather_evaluate(conformal_solution, true, false);
          diff_conf_in.reinit(face);
          diff_conf_in.gather_evaluate(diff_conf_solution, true, false);
          diff_conf_out.reinit(face);
          diff_conf_out.gather_evaluate(diff_conf_solution, true, false);

          for (unsigned int q=0; q<new_conf_in.n_q_points; ++q){
            new_conf_in.submit_value(-exp_f.eval(4.*conf_in.get_value(q))*(
              0.5*(diff_conf_in.get_value(q) + diff_conf_out.get_value(q))
              - tau_val*(conf_in.get_value(q)-conf_out.get_value(q))
              - conf_in.get_value(q)
            )*new_conf_in.get_normal_vector(q)
            ,q);
            new_conf_out.submit_value(exp_f.eval(4.*conf_out.get_value(q))*(
              0.5*(diff_conf_in.get_value(q) + diff_conf_out.get_value(q))
              - tau_val*(conf_in.get_value(q)-conf_out.get_value(q))
              - conf_out.get_value(q)
            )*new_conf_in.get_normal_vector(q)
            ,q);
          }
          new_conf_in.integrate_scatter(EvaluationFlags::values,dst);
          new_conf_out.integrate_scatter(EvaluationFlags::values,dst);
        }
      }

template<int dim>
void IC_Solver<dim>::IC_boundary_apply(const MatrixFree<dim, double> &                   data,
            LinearAlgebra::distributed::Vector<double> &      dst,
            const LinearAlgebra::distributed::Vector<double> &,
            const std::pair<unsigned int, unsigned int> &face_range) const
{/*
  FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> phi_in(data,true,0,1);
  FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> psi_in(data,true,0,1);

  for (unsigned int face = face_range.first; face < face_range.second; ++face){
    phi_in.reinit(face);
    phi_in.gather_evaluate(phi_solution, true, false);
    psi_in.reinit(face);
    psi_in.gather_evaluate(psi_solution, true, false);

    for (unsigned int q=0; q<psi_in.n_q_points; ++q){
      psi_in.submit_value(0.,q);
    }
    psi_in.integrate_scatter(EvaluationFlags::values,dst);
  }*/

}





    // @sect4{IC_Solver::solve}

    // The solution process is similar as in step-16. We start with the setup of
    // the transfer. For LinearAlgebra::distributed::Vector, there is a very
    // fast transfer class called MGTransferMatrixFree that does the
    // interpolation between the grid levels with the same fast sum
    // factorization kernels that get also used in FEEvaluation.
    template <int dim>
    void IC_Solver<dim>::solve(const unsigned int &nlevels, const DoFHandler<dim> &dof_handler_DG)
    {
      //std::cout << "K solver" << std::endl;
      general_solver(nlevels,dof_handler_DG, constraints_DG, mg_constrained_dofs_DG,
         IC_matrix, IC_matrices, new_conformal_solution, IC_rhs);
      //std::cout << "U solver" << std::endl;
    }

    // @sect4{IC_Solver::run}

    // The function that runs the program is very similar to the one in
    // step-16. We do few refinement steps in 3D compared to 2D, but that's
    // it.
    //
    // Before we run the program, we output some information about the detected
    // vectorization level as discussed in the introduction.
    template <int dim>
    void IC_Solver<dim>::run()
    {
      {
        const unsigned int n_vect_doubles = VectorizedArray<double>::size();
        const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

        pcout << "Vectorization over " << n_vect_doubles
              << " doubles = " << n_vect_bits << " bits ("
              << Utilities::System::get_current_vectorization_level() << ')'
              << std::endl;
      }


      setup_system();
      assemble_rhs();
      solve();
    }
      //pcout << std::endl;
}//namespace Brill_Evolution
