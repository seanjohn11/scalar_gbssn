namespace Scalar_Evolution
{
  using namespace dealii;

  template <int dim, typename SystemMatrixType_name, typename LevelMatrixType_name>
  void general_UK_solver(const unsigned int &nlevels,const DoFHandler<dim> &dof_handler,
     const AffineConstraints<double> &constrain, MGConstrainedDoFs &mg_constrain,
     SystemMatrixType_name &syst_matrix, MGLevelObject<LevelMatrixType_name> &mg_matrice,
     LinearAlgebra::distributed::Vector<double> &sol, LinearAlgebra::distributed::Vector<double> &rhs)
  {
        Timer                            time;
        MGTransferMatrixFree<dim, float> mg_transfer(mg_constrain);
        mg_transfer.build(dof_handler);


        using SmootherType =
          PreconditionChebyshev<LevelMatrixType_name,
                                LinearAlgebra::distributed::Vector<float>>;
        mg::SmootherRelaxation<SmootherType,
                               LinearAlgebra::distributed::Vector<float>>
                                                             mg_smoother;
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        smoother_data.resize(0, nlevels - 1);
        for (unsigned int level = 0; level < nlevels;
             ++level)
          {
            if (level > 0)
              {
                smoother_data[level].smoothing_range     = 15.;
                smoother_data[level].degree              = 5;
                smoother_data[level].eig_cg_n_iterations = 10;
              }
            else
              {
                smoother_data[0].smoothing_range = 1e-3;
                smoother_data[0].degree          = numbers::invalid_unsigned_int;
                smoother_data[0].eig_cg_n_iterations = mg_matrice[0].m()+3;
              }
            mg_matrice[level].compute_diagonal();
            smoother_data[level].preconditioner =
              mg_matrice[level].get_matrix_diagonal_inverse();
          }
        mg_smoother.initialize(mg_matrice, smoother_data);

        MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>>
          mg_coarse;
        mg_coarse.initialize(mg_smoother);

        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(
          mg_matrice);

        MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType_name>>
          mg_interface_matrices;
        mg_interface_matrices.resize(0, nlevels - 1);
        for (unsigned int level = 0; level < nlevels;
             ++level)
          mg_interface_matrices[level].initialize(mg_matrice[level]);
        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(
          mg_interface_matrices);

        Multigrid<LinearAlgebra::distributed::Vector<float>> mg(
          mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
        mg.set_edge_matrices(mg_interface, mg_interface);

        PreconditionMG<dim,
                       LinearAlgebra::distributed::Vector<float>,
                       MGTransferMatrixFree<dim, float>>
          preconditioner(dof_handler, mg, mg_transfer);

        // The setup of the multigrid routines is quite easy and one cannot see
        // any difference in the solve process as compared to step-16. All the
        // magic is hidden behind the implementation of the LaplaceOperator::vmult
        // operation. Note that we print out the solve time and the accumulated
        // setup time through standard out, i.e., in any case, whereas detailed
        // times for the setup operations are only printed in case the flag for
        // detail_times in the constructor is changed.

        SolverControl solver_control(100, 1e-12 * rhs.l2_norm());
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        //SolverBicgstab<LinearAlgebra::distributed::Vector<double>> bicgstab(solver_control);
        //SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);
        //setup_time += time.wall_time();


        time.reset();
        time.start();
        //std::cout << "I made it to 1" << std::endl;
        constrain.set_zero(sol);
        //constrain.set_zero(rhs);

        cg.solve(syst_matrix, sol, rhs, preconditioner);
        //bicgstab.solve(syst_matrix, sol, rhs, preconditioner);
        //gmres.solve(syst_matrix, sol, rhs, preconditioner);
        //std::cout << "I made it to 3" << std::endl;
        constrain.distribute(sol);

        //std::cout << "Time solve (" << solver_control.last_step() <<
        //" iterations)\n" << std::endl;
  }





  template <int dim>
    class Diff_IC_Solver
    {
    public:
      Diff_IC_Solver();
      void run();


      void setup_system(const unsigned int &nlevels,
              const MappingQ<dim> &mapping,
              DoFHandler<dim> &dof_handler_DG,
              const FESystem<dim>   &fe_DG);
      void assemble_rhs();
      void solve(const unsigned int &nlevels, const DoFHandler<dim> &dof_handler_DG);

    private:
      AffineConstraints<double> constraints_DG;
      using SystemMatrixType =
          Diff_IC_Operator<dim, fe_degree, double>;
      SystemMatrixType Diff_matrix;

      MGConstrainedDoFs mg_constrained_dofs_DG;
      using LevelMatrixType = Diff_IC_Operator<dim, fe_degree, float>;
      MGLevelObject<LevelMatrixType> Diff_matrices;

      LinearAlgebra::distributed::Vector<double> conf_rhs;
      LinearAlgebra::distributed::Vector<double> new_conf_rhs;

      double             setup_time;
      ConditionalOStream pcout;
      ConditionalOStream time_details;

      void face_conf_apply(const MatrixFree<dim, double> &                   data,
                  LinearAlgebra::distributed::Vector<double> &      dst,
                  const LinearAlgebra::distributed::Vector<double> &src,
                  const std::pair<unsigned int, unsigned int> &face_range) const;

      void face_new_conf_apply(const MatrixFree<dim, double> &                   data,
                  LinearAlgebra::distributed::Vector<double> &      dst,
                  const LinearAlgebra::distributed::Vector<double> &src,
                  const std::pair<unsigned int, unsigned int> &face_range) const;

      void conf_cell_apply(const MatrixFree<dim, double> &                   data,
                  LinearAlgebra::distributed::Vector<double> &      dst,
                  const LinearAlgebra::distributed::Vector<double> &src,
                  const std::pair<unsigned int, unsigned int> &cell_range) const;

      void new_conf_cell_apply(const MatrixFree<dim, double> &                   data,
                LinearAlgebra::distributed::Vector<double> &      dst,
                const LinearAlgebra::distributed::Vector<double> &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

      void conf_boundary_apply(const MatrixFree<dim, double> &                   data,
                  LinearAlgebra::distributed::Vector<double> &      dst,
                  const LinearAlgebra::distributed::Vector<double> &src,
                  const std::pair<unsigned int, unsigned int> &face_range) const;

      void new_conf_boundary_apply(const MatrixFree<dim, double> &                   data,
                  LinearAlgebra::distributed::Vector<double> &      dst,
                  const LinearAlgebra::distributed::Vector<double> &src,
                  const std::pair<unsigned int, unsigned int> &face_range) const;
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
    Diff_IC_Solver<dim>::Diff_IC_Solver()
      : setup_time(0.)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      // The Diff_IC_Solver class holds an additional output stream that
      // collects detailed timings about the setup phase. This stream, called
      // time_details, is disabled by default through the @p false argument
      // specified here. For detailed timings, removing the @p false argument
      // prints all the details.
      , time_details(std::cout,
                     false &&
                       Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {}



    // @sect4{Diff_IC_Solver::setup_system}

    template <int dim>
    void Diff_IC_Solver<dim>::setup_system(const unsigned int &nlevels,
            const MappingQ<dim> &mapping,
            DoFHandler<dim> &dof_handler_DG,
            const FESystem<dim>   &fe_DG)
    {
      Timer time;
      setup_time = 0;

      Diff_matrix.clear();
      Diff_matrices.clear_elements();

      dof_handler_DG.distribute_dofs(fe_DG);
      dof_handler_DG.distribute_mg_dofs();

      const IndexSet locally_relevant_dofs_DG =
        DoFTools::extract_locally_relevant_dofs(dof_handler_DG);

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
        Diff_matrix.initialize(system_mf_storage,{0});

        Diff_matrix.initialize_dof_vector(conf_rhs);
        Diff_matrix.initialize_dof_vector(new_conf_rhs);
        Diff_matrix.initialize_dof_vector(diff_conf_solution);
      }

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
      Diff_matrices.resize(0, nlevels - 1);

      mg_constrained_dofs_DG.initialize(dof_handler_DG);


      for (unsigned int level = 0; level < nlevels; ++level)
        {
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
                                      QGauss<dim>(fe_degree + 1),
                                      additional_data);

          Diff_matrices[level].initialize(mg_mf_storage_level,
                                        mg_constrained_dofs_DG,
                                        level,
                                        {0});

        }
      setup_time += time.wall_time();
      time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time()
                   << "s/" << time.wall_time() << 's' << std::endl;
    }



    // @sect4{Diff_IC_Solver::assemble_rhs}

    // The assemble function is very simple since all we have to do is to
    // assemble the right hand side. Thanks to FEEvaluation and all the data
    // cached in the MatrixFree class, which we query from
    // MatrixFreeOperators::Base, this can be done in a few lines. Since this
    // call is not wrapped into a MatrixFree::cell_loop (which would be an
    // alternative), we must not forget to call compress() at the end of the
    // assembly to send all the contributions of the right hand side to the
    // owner of the respective degree of freedom.
    template <int dim>
    void Diff_IC_Solver<dim>::assemble_rhs()
    {
      /*drho_psi_rhs = 0;
      dz_psi_rhs = 0;
      drho_s_rhs = 0;
      dz_s_rhs = 0;
      drho_alpha_rhs = 0;
      dz_alpha_rhs = 0;*/

      this->Diff_matrix.get_matrix_free()->loop(&Diff_IC_Solver::conf_cell_apply,
                       &Diff_IC_Solver::face_conf_apply,
                       &Diff_IC_Solver::conf_boundary_apply,
                       this, conf_rhs, diff_conf_solution,
                       true
                       ,MatrixFree<dim, double>::DataAccessOnFaces::values
                       ,MatrixFree<dim, double>::DataAccessOnFaces::values
                     );

       /*this->Diff_matrix.get_matrix_free()->loop(&Diff_IC_Solver::new_conf_cell_apply,
                        &Diff_IC_Solver::face_new_conf_apply,
                        &Diff_IC_Solver::new_conf_boundary_apply,
                        this, new_conf_rhs, new_conformal_solution,
                        true
                        ,MatrixFree<dim, double>::DataAccessOnFaces::values
                        ,MatrixFree<dim, double>::DataAccessOnFaces::values
                      );*/
    }

    template <int dim>
    void Diff_IC_Solver<dim>::conf_cell_apply(const MatrixFree<dim, double> &                   data,
                LinearAlgebra::distributed::Vector<double> &      dst,
                const LinearAlgebra::distributed::Vector<double> &,
                const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff(data,0,1);
      FEEvaluation<dim, fe_degree, alt_q_points, 1, Number> conf(data,0,1);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
           {
             diff.reinit(cell);
             //diff.gather_evaluate(V_Diff_Solution,false,true);
             conf.reinit(cell);
             conf.gather_evaluate(conformal_solution,false,true);
             for (unsigned int q=0; q < diff.n_q_points; ++q){
               diff.submit_value(conf.get_gradient(q),q);
             }
             diff.integrate(EvaluationFlags::values);
             diff.distribute_local_to_global(dst);

           }
      }

      template <int dim>
      void Diff_IC_Solver<dim>::new_conf_cell_apply(const MatrixFree<dim, double> &                   data,
                  LinearAlgebra::distributed::Vector<double> &      dst,
                  const LinearAlgebra::distributed::Vector<double> &,
                  const std::pair<unsigned int, unsigned int> &cell_range) const
      {
        FEEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff(data,0,1);
        FEEvaluation<dim, fe_degree, alt_q_points, 1, Number> new_conf(data,0,1);


        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
             {
               diff.reinit(cell);
               //diff.gather_evaluate(V_Diff_Solution,false,true);
               new_conf.reinit(cell);
               new_conf.gather_evaluate(new_conformal_solution,false,true);
               for (unsigned int q=0; q < diff.n_q_points; ++q){
                 diff.submit_value(new_conf.get_gradient(q),q);
               }
               diff.integrate_scatter(true,false,dst);

             }
        }

      template <int dim>
      void Diff_IC_Solver<dim>::face_conf_apply(const MatrixFree<dim, double> &data  ,
                  LinearAlgebra::distributed::Vector<double> &dst      ,
                  const LinearAlgebra::distributed::Vector<double> &,
                  const std::pair<unsigned int, unsigned int> &face_range) const
      {
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff_in(data,true,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff_out(data,false,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> conf_in(data,true,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> conf_out(data,false,0,1);

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            diff_in.reinit(face);
            //v_diff.gather_evaluate(V_Diff_Solution,false,true);
            diff_out.reinit(face);
            conf_in.reinit(face);
            conf_in.gather_evaluate(conformal_solution,true,false);
            conf_out.reinit(face);
            conf_out.gather_evaluate(conformal_solution,true,false);

              for (unsigned int q = 0; q < diff_in.n_q_points; ++q)
                {
                  diff_in.submit_value((0.5*(conf_in.get_value(q) + conf_out.get_value(q))-conf_in.get_value(q))*diff_in.get_normal_vector(q),q);
                  diff_out.submit_value(-(0.5*(conf_in.get_value(q) + conf_out.get_value(q))-conf_out.get_value(q))*diff_in.get_normal_vector(q),q);
                }
            diff_in.integrate_scatter(EvaluationFlags::values, dst);
            diff_out.integrate_scatter(EvaluationFlags::values, dst);
        }
      }

      template <int dim>
      void Diff_IC_Solver<dim>::face_new_conf_apply(const MatrixFree<dim, double> &data  ,
                  LinearAlgebra::distributed::Vector<double> &dst      ,
                  const LinearAlgebra::distributed::Vector<double> &,
                  const std::pair<unsigned int, unsigned int> &face_range) const
      {
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff_in(data,true,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff_out(data,false,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> new_conf_in(data,true,0,1);
        FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> new_conf_out(data,false,0,1);

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            diff_in.reinit(face);
            //v_diff.gather_evaluate(V_Diff_Solution,false,true);
            diff_out.reinit(face);
            new_conf_in.reinit(face);
            new_conf_in.gather_evaluate(new_conformal_solution,true,false);
            new_conf_out.reinit(face);
            new_conf_out.gather_evaluate(new_conformal_solution,true,false);

              for (unsigned int q = 0; q < diff_in.n_q_points; ++q)
                {
                  diff_in.submit_value((0.5*(new_conf_in.get_value(q) + new_conf_out.get_value(q))-new_conf_in.get_value(q))*diff_in.get_normal_vector(q),q);
                  diff_out.submit_value(-(0.5*(new_conf_in.get_value(q) + new_conf_out.get_value(q))-new_conf_out.get_value(q))*diff_in.get_normal_vector(q),q);
                }
            diff_in.integrate_scatter(EvaluationFlags::values, dst);
            diff_out.integrate_scatter(EvaluationFlags::values, dst);
        }
      }

template<int dim>
void Diff_IC_Solver<dim>::conf_boundary_apply(const MatrixFree<dim, double> &                   /*data*/,
            LinearAlgebra::distributed::Vector<double> &      /*dst*/,
            const LinearAlgebra::distributed::Vector<double> &,
            const std::pair<unsigned int, unsigned int> &/*face_range*/) const
{/*
  FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff(data,true,0,1);
  FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> conf(data,true,0,1);

  for (unsigned int face = face_range.first; face < face_range.second; ++face)
    {
      diff.reinit(face);
      //v_diff.gather_evaluate(V_Diff_Solution,false,true);
      conf.reinit(face);
      conf.gather_evaluate(conformal_solution,true,false);

      const auto boundary_id = data.get_boundary_id(face);
        for (unsigned int q = 0; q < diff.n_q_points; ++q)
          {
            if (boundary_id == 3){
              diff.submit_value(0.,q);
            }
            else if (boundary_id == 0){
              diff.submit_value(0.,q);
            }
            else if (boundary_id == 1){
              diff.submit_value(0.,q);
            }
            else if (boundary_id == 4){
              diff.submit_value(0.,q);
            }
          }
      diff.integrate_scatter(EvaluationFlags::values, dst);
  }*/
}

template<int dim>
void Diff_IC_Solver<dim>::new_conf_boundary_apply(const MatrixFree<dim, double> &                   /*data*/,
            LinearAlgebra::distributed::Vector<double> &      /*dst*/,
            const LinearAlgebra::distributed::Vector<double> &,
            const std::pair<unsigned int, unsigned int> &/*face_range*/) const
{/*
  FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> diff(data,true,0,1);
  FEFaceEvaluation<dim, fe_degree, alt_q_points, 1, Number> new_conf(data,true,0,1);

  for (unsigned int face = face_range.first; face < face_range.second; ++face)
    {

      diff.reinit(face);
      //v_diff.gather_evaluate(V_Diff_Solution,false,true);
      new_conf.reinit(face);
      new_conf.gather_evaluate(new_conf_solution,true,false);

      const auto boundary_id = data.get_boundary_id(face);
        for (unsigned int q = 0; q < diff.n_q_points; ++q)
          {
            if (boundary_id == 3){
              diff.submit_value(0.,q);
            }
            else if (boundary_id == 0){
              diff.submit_value(0.,q);
            }
            else if (boundary_id == 1){
              diff.submit_value(0.,q);
            }
            else if (boundary_id == 4){
              diff.submit_value(0
      beta_z.reinit(face);
      beta_z.gather_evaluate(beta_z_solution,true,false);
      alpha.reinit(face);
      alpha.gather_evaluate(alpha_solution,true,false);.,q);
            }
          }
      diff.integrate_scatter(EvaluationFlags::values, dst);
  }*/
}





    // @sect4{Diff_IC_Solver::solve}

    // The solution process is similar as in step-16. We start with the setup of
    // the transfer. For LinearAlgebra::distributed::Vector, there is a very
    // fast transfer class called MGTransferMatrixFree that does the
    // interpolation between the grid levels with the same fast sum
    // factorization kernels that get also used in FEEvaluation.
    template <int dim>
    void Diff_IC_Solver<dim>::solve(const unsigned int &nlevels, const DoFHandler<dim> &dof_handler_DG)
    {
      //std::cout << "K solver" << std::endl;
      general_UK_solver(nlevels,dof_handler_DG, constraints_DG, mg_constrained_dofs_DG,
         Diff_matrix, Diff_matrices, diff_conf_solution, conf_rhs);
      //std::cout << "U solver" << std::endl;
      /*general_UK_solver(nlevels,dof_handler_DG, constraints_DG, mg_constrained_dofs_DG,
         Diff_matrix, Diff_matrices, diff_new_conf_solution, new_conf_rhs);*/
    }

    // @sect4{Diff_IC_Solver::run}

    // The function that runs the program is very similar to the one in
    // step-16. We do few refinement steps in 3D compared to 2D, but that's
    // it.
    //
    // Before we run the program, we output some information about the detected
    // vectorization level as discussed in the introduction.
    template <int dim>
    void Diff_IC_Solver<dim>::run()
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
