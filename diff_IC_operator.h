namespace Scalar_Evolution
{
  using namespace dealii;

  template <int dim, int fe_degree, typename number>
  class Diff_IC_Operator
    : public MatrixFreeOperators::
        Base<dim, LinearAlgebra::distributed::Vector<number>>
  {
  public:
    using value_type = number;

    Diff_IC_Operator();

    void clear() override;

    virtual void compute_diagonal() override;

  private:
    virtual void apply_add(
      LinearAlgebra::distributed::Vector<number> &      dst,
      const LinearAlgebra::distributed::Vector<number> &src) const override;

    void
    cell_apply(const MatrixFree<dim, number> &                   data,
                LinearAlgebra::distributed::Vector<number> &      dst,
                const LinearAlgebra::distributed::Vector<number> &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    void cell_diagonal(
      const MatrixFree<dim, number> &              data,
      LinearAlgebra::distributed::Vector<number> & dst,
      const unsigned int &                         dummy,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    face_apply(const MatrixFree<dim, number> &                   data,
                LinearAlgebra::distributed::Vector<number> &      dst,
                const LinearAlgebra::distributed::Vector<number> &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    boundary_apply(const MatrixFree<dim, number> &                   data,
                LinearAlgebra::distributed::Vector<number> &      dst,
                const LinearAlgebra::distributed::Vector<number> &src,
                const std::pair<unsigned int, unsigned int> &face_range) const;

  };



  // This is the constructor of the @p Diff_IC_Operator class. All it does is
  // to call the default constructor of the base class
  // MatrixFreeOperators::Base, which in turn is based on the Subscriptor
  // class that asserts that this class is not accessed after going out of scope
  // e.g. in a preconditioner.
  template <int dim, int fe_degree, typename number>
  Diff_IC_Operator<dim, fe_degree, number>::Diff_IC_Operator()
    : MatrixFreeOperators::Base<dim,
                                LinearAlgebra::distributed::Vector<number>>()
  {}



  template <int dim, int fe_degree, typename number>
  void Diff_IC_Operator<dim, fe_degree, number>::clear()
  {
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::
      clear();
  }

  // @sect4{Local evaluation of V_Diff operator}

  template <int dim, int fe_degree, typename number>
  void Diff_IC_Operator<dim, fe_degree, number>::cell_apply(
    const MatrixFree<dim, number> &                   data,
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &s_diff_vec,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> s_diff(data,0);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        s_diff.reinit(cell);
        s_diff.gather_evaluate(s_diff_vec,true,false);

        for (unsigned int q = 0; q < s_diff.n_q_points; ++q){
          //VectorizedArray<number> rho_val = std::abs(V_Diff.quadrature_point(q)[0]);
          s_diff.submit_value(s_diff.get_value(q),q);}
        s_diff.integrate_scatter(true,false,dst);
      }
  }

  template <int dim, int fe_degree, typename number>
  void Diff_IC_Operator<dim, fe_degree, number>::face_apply(
    const MatrixFree<dim, number> &                   data,
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &diff_vec,
    const std::pair<unsigned int, unsigned int> &face_range     ) const
  {
  }

  template <int dim, int fe_degree, typename number>
  void Diff_IC_Operator<dim, fe_degree, number>::boundary_apply(
    const MatrixFree<dim, number> &                  ,
    LinearAlgebra::distributed::Vector<number> &      ,
    const LinearAlgebra::distributed::Vector<number> &,
    const std::pair<unsigned int, unsigned int> &     ) const
  {
  }

  template <int dim, int fe_degree, typename number>
  void Diff_IC_Operator<dim, fe_degree, number>::apply_add(
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &src) const
  {
  this->data->cell_loop(&Diff_IC_Operator::cell_apply,
                   //&Diff_IC_Operator::face_apply,
                   //&Diff_IC_Operator::boundary_apply,
                   this, dst, src,
                   true
                   //,MatrixFree<dim, number>::DataAccessOnFaces::values
                   //,MatrixFree<dim, number>::DataAccessOnFaces::values
                 );
  }

  template <int dim, int fe_degree, typename number>
  void Diff_IC_Operator<dim, fe_degree, number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal,0);
    unsigned int dummy = 0;

    this->data->cell_loop(&Diff_IC_Operator::cell_diagonal,
                     //&Diff_IC_Operator::face_diagonal,
                     //&Diff_IC_Operator::boundary_diagonal,
                          this,
                          inverse_diagonal,
                          dummy,
                        true
                      //  ,MatrixFree<dim, number>::DataAccessOnFaces::values,MatrixFree<dim, number>::DataAccessOnFaces::values
                      );

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
      {
        //std::cout << inverse_diagonal.local_element(i) << std::endl;
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        inverse_diagonal.local_element(i) =
          1. / inverse_diagonal.local_element(i);
      }
  }



  // In the local compute loop, we compute the diagonal by a loop over all
  // columns in the local matrix and putting the entry 1 in the <i>i</i>th
  // slot and a zero entry in all other slots, i.e., we apply the cell-wise
  // differential operator on one unit vector at a time. The inner part
  // invoking FEEvaluation::evaluate, the loop over quadrature points, and
  // FEEvalution::integrate, is exactly the same as in the stiffness_apply
  // function. Afterwards, we pick out the <i>i</i>th entry of the local
  // result and put it to a temporary storage (as we overwrite all entries in
  // the array behind FEEvaluation::get_dof_value() with the next loop
  // iteration). Finally, the temporary storage is written to the destination
  // vector. Note how we use FEEvaluation::get_dof_value() and
  // FEEvaluation::submit_dof_value() to read and write to the data field that
  // FEEvaluation uses for the integration on the one hand and writes into the
  // global vector on the other hand.
  //
  // Given that we are only interested in the matrix diagonal, we simply throw
  // away all other entries of the local matrix that have been computed along
  // the way. While it might seem wasteful to compute the complete cell matrix
  // and then throw away everything but the diagonal, the integration are so
  // efficient that the computation does not take too much time. Note that the
  // complexity of operator evaluation per element is $\mathcal
  // O((p+1)^{d+1})$ for polynomial degree $k$, so computing the whole matrix
  // costs us $\mathcal O((p+1)^{2d+1})$ operations, not too far away from
  // $\mathcal O((p+1)^{2d})$ complexity for computing the diagonal with
  // FEValues. Since FEEvaluation is also considerably faster due to
  // vectorization and other optimizations, the diagonal computation with this
  // function is actually the fastest (simple) variant. (It would be possible
  // to compute the diagonal with sum factorization techniques in $\mathcal
  // O((p+1)^{d+1})$ operations involving specifically adapted
  // kernels&mdash;but since such kernels are only useful in that particular
  // context and the diagonal computation is typically not on the critical
  // path, they have not been implemented in deal.II.)
  //
  // Note that the code that calls distribute_local_to_global on the vector to
  // accumulate the diagonal entries into the global matrix has some
  // limitations. For operators with hanging node constraints that distribute
  // an integral contribution of a constrained DoF to several other entries
  // inside the distribute_local_to_global call, the vector interface used
  // here does not exactly compute the diagonal entries, but lumps some
  // contributions located on the diagonal of the local matrix that would end
  // up in a off-diagonal position of the global matrix to the diagonal. The
  // result is correct up to discretization accuracy as explained in <a
  // href="http://dx.doi.org/10.4208/cicp.101214.021015a">Kormann (2016),
  // section 5.3</a>, but not mathematically equal. In this tutorial program,
  // no harm can happen because the diagonal is only used for the multigrid
  // level matrices where no hanging node constraints appear.
  template <int dim, int fe_degree, typename number>
  void Diff_IC_Operator<dim, fe_degree, number>::cell_diagonal(
    const MatrixFree<dim, number> &             data,
    LinearAlgebra::distributed::Vector<number> &dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    // Got lazy. All the V_Diff should be s_diff. Its an inconsequential change though
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> V_Diff(data,0);

    AlignedVector<VectorizedArray<number>> diagonal(V_Diff.dofs_per_cell);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        V_Diff.reinit(cell);

        for (unsigned int i = 0; i < V_Diff.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < V_Diff.dofs_per_cell; ++j){
              V_Diff.submit_dof_value(VectorizedArray<number>(), j);

            }
            V_Diff.submit_dof_value(make_vectorized_array<number>(1.), i);


            V_Diff.evaluate(true, false);

            for (unsigned int q = 0; q < V_Diff.n_q_points; ++q){
              //VectorizedArray<number> rho_val = std::abs(V_Diff.quadrature_point(q)[0]);
              V_Diff.submit_value(V_Diff.get_value(q),q);}
            V_Diff.integrate(true,false);
            diagonal[i] = V_Diff.get_dof_value(i);
          }
        for (unsigned int i = 0; i < V_Diff.dofs_per_cell; ++i)
          V_Diff.submit_dof_value(diagonal[i], i);
        V_Diff.distribute_local_to_global(dst);
      }
  }

}//namespace Brill_Evolution
