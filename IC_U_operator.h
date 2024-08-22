namespace Scalar_Evolution
{
  using namespace dealii;


  template <int dim, int degree, int n_points_1d>
  class UOperator
  {
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    UOperator(TimerOutput &timer_output);

    void reinit(const Mapping<dim> &   mapping,
    const DoFHandler<dim>  &dof_handlers,
    const AffineConstraints<double>  &constraints);

    void apply(const double                                      current_time,
               const LinearAlgebra::distributed::Vector<Number> &src,
               LinearAlgebra::distributed::Vector<Number> &      dst) const;

    void
    perform_stage(const Number cur_time,
                  const Number factor_solution,
                  const Number factor_ai,
                  const LinearAlgebra::distributed::Vector<Number> &current_ri,
                  LinearAlgebra::distributed::Vector<Number> &      vec_ki,
                  LinearAlgebra::distributed::Vector<Number> &      solution,
                  LinearAlgebra::distributed::Vector<Number> &next_ri) const;

    void
    initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector) const;

    MatrixFree<dim, Number> data;
  private:


    TimerOutput &timer;

    void local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void local_apply_cell(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void local_apply_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     face_range) const;

    void local_apply_boundary_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     face_range) const;
  };



  template <int dim, int degree, int n_points_1d>
  UOperator<dim, degree, n_points_1d>::UOperator(TimerOutput &timer)
    : timer(timer)
  {}



  // For the initialization of the Psi operator, we set up the MatrixFree
  // variable contained in the class. This can be done given a mapping to
  // describe possible curved boundaries as well as a DoFHandler object
  // describing the degrees of freedom. Since we use a discontinuous Galerkin
  // discretization in this tutorial program where no constraints are imposed
  // strongly on the solution field, we do not need to pass in an
  // AffineConstraints object and rather use a dummy for the
  // construction.
  template <int dim, int degree, int n_points_1d>
  void UOperator<dim, degree, n_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const DoFHandler<dim> &dof_handler_DG,
    const AffineConstraints<double>  &constraint_DG)
    // Probably create the vector below of dof handlers outside of the classes
    // and then change the second argument of the functino so this is handed over.
    // Show make it so duplicates are not made all over the place
    //Current Plan for numbering
    // 0 Psi, 1 S, 2 Kz_rho, 3 U, 4 W, 5 alpha, 6 Beta_rho, 7 Beta_z
  {/*
    const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler_DG, &dof_handler_CG};
    const AffineConstraints<double>            constraints_DG;
    const AffineConstraints<double>            constraints_DG_odd;
    const AffineConstraints<double>            contraints;
    const AffineConstraints<double>            contraints_rho;
    const std::vector<const AffineConstraints<double> *> constraints = {&dummy,&dummy_2};*/
    const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler_DG};
    const std::vector<const AffineConstraints<double> *> constraints = {&constraint_DG};
    const std::vector<Quadrature<dim>> quadratures = {QGauss<dim>(alt_q_points),
                                                      QGauss<dim>(n_q_points_1d)};

    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points |
       update_values);
    additional_data.mapping_update_flags_inner_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors | update_gradients |
       update_values);
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::none;

    data.reinit(
      mapping, dof_handlers, constraints, quadratures, additional_data);
  }



  template <int dim, int degree, int n_points_1d>
  void UOperator<dim, degree, n_points_1d>::initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector) const
  {
    data.initialize_dof_vector(vector,0);
  }

  // @sect4{Local evaluators}

  // This is where the DGFEM formulation actually are implimented.
  // The local_apply_cell section are for integrals over the entire cell,
  // local_apply_face is for interior faces and local_apply_boundary_face is
  // for boundary integrals.
  // Final numbers in the FEEvalutaion variables are important as they choose
  // the dof_handler that is used and therefore the variable itself.

  template <int dim, int degree, int n_points_1d>
  void UOperator<dim, degree, n_points_1d>::local_apply_cell(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> & conf_vec,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, alt_q_points, 1, Number> conf(data,0);
    FEEvaluation<dim, degree, alt_q_points, 1, Number> v(data,0);


    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        conf.reinit(cell);
        conf.gather_evaluate(conf_vec, EvaluationFlags::values);
        v.reinit(cell);
        v.gather_evaluate(xi_solution, EvaluationFlags::values);
        for (unsigned int q = 0; q < conf.n_q_points; ++q)
          {
            conf.submit_value(v.get_value(q) - eta_val*conf.get_value(q)
            ,q);
          }
        conf.integrate_scatter(EvaluationFlags::values,dst);
      }
  }


  template <int dim, int degree, int n_points_1d>
  void UOperator<dim, degree, n_points_1d>::local_apply_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      /*dst*/,
    const LinearAlgebra::distributed::Vector<Number> &/*conf_vec*/,
    const std::pair<unsigned int, unsigned int> &     /*face_range*/) const
  {/*
    // The booleans true/false in the namings are defining whether the value is
    // from the interior/exterior relative to the normal vector provided.
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> conf_in(data, true,0);
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> conf_out(data, false, 0);
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> diff_conf_in(data,true, 0);
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> diff_conf_out(data,false, 0);


    //Need to change the normal_tester so it is a "beta" function but I believe
    // Possible solution implemented

    exp_func<dim> exp_f;


    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        conf_in.reinit(face);
        conf_in.gather_evaluate(conf_vec, EvaluationFlags::values);
        conf_out.reinit(face);
        conf_out.gather_evaluate(conf_vec, EvaluationFlags::values);
        diff_conf_in.reinit(face);
        diff_conf_in.gather_evaluate(diff_conf_solution, EvaluationFlags::values);
        diff_conf_out.reinit(face);
        diff_conf_out.gather_evaluate(diff_conf_solution, EvaluationFlags::values);

        for (unsigned int q = 0; q < conf_in.n_q_points; ++q)
          {
            const VectorizedArray<Number> diff_conf_avg =
                  (diff_conf_in.get_value(q) + diff_conf_out.get_value(q));
            const VectorizedArray<Number> conf_jump =
                  (conf_in.get_value(q) - conf_out.get_value(q));

            conf_in.submit_value(exp_f.eval(conf_in.get_value(q))*(0.5*diff_conf_avg - tau_val*conf_jump - diff_conf_in.get_value(q)),q);
            conf_out.submit_value(-exp_f.eval(conf_out.get_value(q))*(0.5*diff_conf_avg - tau_val*conf_jump - diff_conf_out.get_value(q)),q);

          }
        conf_in.integrate_scatter(EvaluationFlags::values, dst);
        conf_out.integrate_scatter(EvaluationFlags::values,dst);

    }*/
  }




  template <int dim, int degree, int n_points_1d>
  void UOperator<dim, degree, n_points_1d>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &conf_vec,
    const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> conf_in(data, true,0);
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> diff_conf_in(data,true, 0);


    //Need to change the normal_tester so it is a "beta" function but I believe
    // Possible solution implemented

    exp_func<dim> exp_f;


    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        conf_in.reinit(face);
        conf_in.gather_evaluate(conf_vec, EvaluationFlags::values);
        diff_conf_in.reinit(face);
        diff_conf_in.gather_evaluate(diff_conf_solution, EvaluationFlags::values);
        const auto boundary_id = data.get_boundary_id(face);
        for (unsigned int q = 0; q < conf_in.n_q_points; ++q)
          {
            if (boundary_id == 0){
              conf_in.submit_value(0.,q);
            }
            else if (boundary_id == 1){
              //conf_in.submit_value(0.,q);
              conf_in.submit_value(-conf_in.get_value(q),q);
            }

          }
        conf_in.integrate_scatter(EvaluationFlags::values, dst);

    }
  }



  // The next function implements the inverse mass matrix operation. The
  // algorithms and rationale have been discussed extensively in the
  // introduction, so we here limit ourselves to the technicalities of the
  // MatrixFreeOperators::CellwiseInverseMassMatrix class. It does similar
  // operations as the forward evaluation of the mass matrix, except with a
  // different interpolation matrix, representing the inverse $S^{-1}$
  // factors. These represent a change of basis from the specified basis (in
  // this case, the Lagrange basis in the points of the Gauss--Lobatto
  // quadrature formula) to the Lagrange basis in the points of the Gauss
  // quadrature formula. In the latter basis, we can apply the inverse of the
  // point-wise `JxW` factor, i.e., the quadrature weight times the
  // determinant of the Jacobian of the mapping from reference to real
  // coordinates. Once this is done, the basis is changed back to the nodal
  // Gauss-Lobatto basis again. All of these operations are done by the
  // `apply()` function below. What we need to provide is the local fields to
  // operate on (which we extract from the global vector by an FEEvaluation
  // object) and write the results back to the destination vector of the mass
  // matrix operation.
  //
  // One thing to note is that we added two integer arguments (that are
  // optional) to the constructor of FEEvaluation, the first being 0
  // (selecting among the DoFHandler in multi-DoFHandler systems; here, we
  // only have one) and the second being 1 to make the quadrature formula
  // selection. As we use the quadrature formula 0 for the over-integration of
  // nonlinear terms, we use the formula 1 with the default $p+1$ (or
  // `fe_degree+1` in terms of the variable name) points for the mass
  // matrix. This leads to square contributions to the mass matrix and ensures
  // exact integration, as explained in the introduction.
  template <int dim, int degree, int n_points_1d>
  void UOperator<dim, degree, n_points_1d>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, n_q_points_1d, 1, Number> psi(data, 0,1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, 1, Number>
      inverse(psi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        psi.reinit(cell);
        psi.read_dof_values(src);

        inverse.apply(psi.begin_dof_values(), psi.begin_dof_values());

        psi.set_dof_values(dst);
      }
  }



  // @sect4{The apply() and related functions}

  // We now come to the function which implements the evaluation of the psi
  // operator as a whole, i.e., $\mathcal M^{-1} \mathcal L(t, \mathbf{w})$,
  // calling into the local evaluators presented above. The steps should be
  // clear from the previous code. One thing to note is that we need to adjust
  // the time in the functions we have associated with the various parts of
  // the boundary, in order to be consistent with the equation in case the
  // boundary data is time-dependent. Then, we call MatrixFree::loop() to
  // perform the cell and face integrals, including the necessary ghost data
  // exchange in the `src` vector. The seventh argument to the function,
  // `true`, specifies that we want to zero the `dst` vector as part of the
  // loop, before we start accumulating integrals into it. This variant is
  // preferred over explicitly calling `dst = 0.;` before the loop as the
  // zeroing operation is done on a subrange of the vector in parts that are
  // written by the integrals nearby. This enhances data locality and allows
  // for caching, saving one roundtrip of vector data to main memory and
  // enhancing performance. The last two arguments to the loop determine which
  // data is exchanged: Since we only access the values of the shape functions
  // one faces, typical of first-order hyperbolic problems, and since we have
  // a nodal basis with nodes at the reference element surface, we only need
  // to exchange those parts. This again saves precious memory bandwidth.
  //
  // Once the spatial operator $\mathcal L$ is applied, we need to make a
  // second round and apply the inverse mass matrix. Here, we call
  // MatrixFree::cell_loop() since only cell integrals appear. The cell loop
  // is cheaper than the full loop as access only goes to the degrees of
  // freedom associated with the locally owned cells, which is simply the
  // locally owned degrees of freedom for DG discretizations. Thus, no ghost
  // exchange is needed here.
  //
  // Around all these functions, we put timer scopes to record the
  // computational time for statistics about the contributions of the various
  // parts.
  template <int dim, int degree, int n_points_1d>
  void UOperator<dim, degree, n_points_1d>::apply(
    const double                                      current_time,
    const LinearAlgebra::distributed::Vector<Number> &src,
    LinearAlgebra::distributed::Vector<Number> &      dst) const
  {
    {
      TimerOutput::Scope t(timer, "apply - integrals");

      data.cell_loop(&UOperator::local_apply_cell,
                //&UOperator::local_apply_face,
                //&UOperator::local_apply_boundary_face,
                this,
                dst,
                src,
                true/*,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values*/);
    }

    {
      TimerOutput::Scope t(timer, "apply - inverse mass");

      data.cell_loop(&UOperator::local_apply_inverse_mass_matrix,
                     this,
                     dst,
                     dst);
    }
  }



  // Let us move to the function that does an entire stage of a Runge--Kutta
  // update. It calls UOperator::apply() followed by some updates
  // to the vectors, namely `next_ri = solution + factor_ai * k_i` and
  // `solution += factor_solution * k_i`. Rather than performing these
  // steps through the vector interfaces, we here present an alternative
  // strategy that is faster on cache-based architectures. As the memory
  // consumed by the vectors is often much larger than what fits into caches,
  // the data has to effectively come from the slow RAM memory. The situation
  // can be improved by loop fusion, i.e., performing both the updates to
  // `next_ki` and `solution` within a single sweep. In that case, we would
  // read the two vectors `rhs` and `solution` and write into `next_ki` and
  // `solution`, compared to at least 4 reads and two writes in the baseline
  // case. Here, we go one step further and perform the loop immediately when
  // the mass matrix inversion has finished on a part of the
  // vector. MatrixFree::cell_loop() provides a mechanism to attach an
  // `std::function` both before the loop over cells first touches a vector
  // entry (which we do not use here, but is e.g. used for zeroing the vector)
  // and a second `std::function` to be called after the loop last touches
  // an entry. The callback is in form of a range over the given vector (in
  // terms of the local index numbering in the MPI universe) that can be
  // addressed by `local_element()` functions.
  //
  // For this second callback, we create a lambda that works on a range and
  // write the respective update on this range. Ideally, we would add the
  // `DEAL_II_OPENMP_SIMD_PRAGMA` before the local loop to suggest to the
  // compiler to SIMD parallelize this loop (which means in practice that we
  // ensure that there is no overlap, also called aliasing, between the index
  // ranges of the pointers we use inside the loops). It turns out that at the
  // time of this writing, GCC 7.2 fails to compile an OpenMP pragma inside a
  // lambda function, so we comment this pragma out below. If your compiler is
  // newer, you should be able to uncomment these lines again.
  //
  // Note that we select a different code path for the last
  // Runge--Kutta stage when we do not need to update the `next_ri`
  // vector. This strategy gives a considerable speedup. Whereas the inverse
  // mass matrix and vector updates take more than 60% of the computational
  // time with default vector updates on a 40-core machine, the percentage is
  // around 35% with the more optimized variant. In other words, this is a
  // speedup of around a third.
  template <int dim, int degree, int n_points_1d>
  void UOperator<dim, degree, n_points_1d>::perform_stage(
    const Number                                      /*current_time*/,
    const Number                                      factor_solution,
    const Number                                      factor_ai,
    const LinearAlgebra::distributed::Vector<Number> &current_ri,
    LinearAlgebra::distributed::Vector<Number> &      vec_ki,
    LinearAlgebra::distributed::Vector<Number> &      solution,
    LinearAlgebra::distributed::Vector<Number> &      next_ri) const
  {
    {
      TimerOutput::Scope t(timer, "rk_stage - integrals L_h");

      data.loop(&UOperator::local_apply_cell,
                &UOperator::local_apply_face,
                &UOperator::local_apply_boundary_face,
                this,
                vec_ki,
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    {
      TimerOutput::Scope t(timer, "rk_stage - inv mass + vec upd");
      data.cell_loop(
        &UOperator::local_apply_inverse_mass_matrix,
        this,
        next_ri,
        vec_ki,
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range) {
          const Number ai = factor_ai;
          const Number bi = factor_solution;
          /*if (ai == Number())
            {
              DEAL_II_OPENMP_SIMD_PRAGMA
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  const Number k_i          = next_ri.local_element(i);
                  const Number sol_i        = solution.local_element(i);
                  solution.local_element(i) = sol_i + bi * k_i;
                }
            }
          else*/
            {
              DEAL_II_OPENMP_SIMD_PRAGMA
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  const Number k_i          = next_ri.local_element(i);

                  const Number sol_i        = solution.local_element(i);
                  solution.local_element(i) = sol_i + bi * k_i;
                  next_ri.local_element(i)  = sol_i + ai * k_i;
                }
            }
        });
    }
  }

} //namespace Brill_Evolution
