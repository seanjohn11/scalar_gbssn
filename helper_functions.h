namespace Scalar_Evolution
{
  using namespace dealii;
  // A function for evaluating the seed function that has been analytically
  // simplified using SageMath



  template <int dim>
  class phi_init:public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p_vectorized,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p_vectorized,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> phi_init<dim>::value(const Point<dim, VectorizedArray<number>> &p_vectorized,
                                 const unsigned int /*component*/) const
  {
    double amp =seed_amp;
    double sig = seed_sigma;
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        double rho = p[0];
        result[v] = amp*exp(-rho*rho/(sig*sig));
      }

    return result;
  }



  template <int dim>
  double phi_init<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    double amp = seed_amp;
    double sig = seed_sigma;
    double rho = p[0];
    return amp*exp(-rho*rho/(sig*sig));
  }




  template <int dim>
  class H_func:public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p_vectorized,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p_vectorized,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> H_func<dim>::value(const Point<dim, VectorizedArray<number>> &p_vectorized,
                                 const unsigned int /*component*/) const
  {
    double amp =seed_amp;
    double sig = seed_sigma;
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        double rho = p[0];
        result[v] = 4.*numbers::PI*amp*amp*rho*rho/(sig*sig*sig*sig)*exp(-2.*rho*rho/(sig*sig));
      }

    return result;
  }



  template <int dim>
  double H_func<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    double amp = seed_amp;
    double sig = seed_sigma;
    double rho = p[0];
    return 4.*numbers::PI*amp*amp*rho*rho/(sig*sig*sig*sig)*exp(-2.*rho*rho/(sig*sig));
  }

  // These are functions for enforcing assymptotic flatness on outer boundaries.

  template <int dim>
  class const_rho_function:public Function<dim>
  {
  public:

    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> const_rho_function<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d)
          q[d] = p[d][v];
        result[v] = q[0]*q[0]/(q[0]*q[0]+q[1]*q[1]);
      }

    return result;
  }



  template <int dim>
  double const_rho_function<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    //return value<double>(p, component);
    double rho = p(0);
    double r_sq = rho*rho + p(1)*p(1);
    return rho*rho/r_sq;
  }




  template <int n_components, int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_components, Number>
    operator*(const Tensor<1, n_components, Tensor<1, dim, Number>> &matrix,
              const Tensor<1, dim, Number> &                         vector)
  {
    Tensor<1, n_components, Number> result;
    for (unsigned int d = 0; d < n_components; ++d)
      result[d] = matrix[d] * vector;
    return result;
  }



  // These functions have been added to try to enforce asymptotic flatness
  // on the outer boundaries of the domain.

  template <int dim>
  class outgoing_rho_function:public Function<dim>
  {
  public:

    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> outgoing_rho_function<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d)
          q[d] = p[d][v];
        result[v] = (q[0]*q[0]+q[1]*q[1])/q[0];
      }

    return result;
  }



  template <int dim>
  double outgoing_rho_function<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    //return value<double>(p, component);
    double rho = p(0);
    double r_sq = (rho*rho + p(1)*p(1))/rho;
    return r_sq;
  }


  template <int dim>
  class outgoing_z_function:public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> outgoing_z_function<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d)
          q[d] = p[d][v];
        result[v] = (q[0]*q[0]+q[1]*q[1])/q[1];
      }

    return result;
  }



  template <int dim>
  double outgoing_z_function<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    //return value<double>(p, component);
    // Returns the alpha factor from Robin Boundary Condition (taken from DOI: 10.1103/PhysRevD.91.044033)
    // Extra rho included for integration reasons
    double rho = p(0);
    double z = p(1);
    double r_sq = rho*rho+z*z;
    return r_sq/z;
  }


  template <int dim>
  class pow_func:public Function<dim>
  {
  public:
    double eval(const double &field,
                         const int &power,
                         const unsigned int component = 0) const;

    template <typename number>
    VectorizedArray<number> eval(const VectorizedArray<number> &field,
                 const int &power,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> pow_func<dim>::eval(const VectorizedArray<number> &field,
                                 const int &power,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        result[v] = std::pow(field[v],power);
      }

    return result;
  }



  template <int dim>
  double pow_func<dim>::eval(const double &field,
                              const int &power,
                                 const unsigned int /*component*/) const
  {
    return std::pow(field,power);
  }


  template <int dim>
  class pow_func_double:public Function<dim>
  {
  public:
    double eval(const double &field,
                         const double &power,
                         const unsigned int component = 0) const;

    template <typename number>
    VectorizedArray<number> eval(const VectorizedArray<number> &field,
                 const double &power,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> pow_func_double<dim>::eval(const VectorizedArray<number> &field,
                                 const double &power,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        result[v] = std::pow(field[v],power);
      }

    return result;
  }



  template <int dim>
  double pow_func_double<dim>::eval(const double &field,
                              const double &power,
                                 const unsigned int /*component*/) const
  {
    return std::pow(field,power);
  }


  template <int dim>
  class exp_func:public Function<dim>
  {
  public:
    virtual double eval(const double &field,
                         const unsigned int component = 0) const;

    template <typename number>
    VectorizedArray<number> eval(const VectorizedArray<number> &field,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> exp_func<dim>::eval(const VectorizedArray<number> &field,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        result[v] = std::exp(field[v]);
      }

    return result;
  }



  template <int dim>
  double exp_func<dim>::eval(const double &field,
                                 const unsigned int /*component*/) const
  {
    return std::exp(field);
  }

template <int dim>
class transferfunc : public Function<dim>
{
public:
  transferfunc(const DoFHandler<dim> &dof_handler) : Function<dim>(), dof_handler_ptr(&dof_handler){}
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;

private:
 const DoFHandler<dim>* dof_handler_ptr;
};

template <int dim>
  double transferfunc<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    const DoFHandler<dim>& dof_handler = *dof_handler_ptr;
    return VectorTools::point_value(dof_handler,psi_solution,p)+1;
  }




  // These function were created to enforce regularity on access.
  // Odd functions require zero on axis and even functions
  // demand a slope value of 0 on axis. Using Garfinkle and Duncan
  // https://arxiv.org/pdf/gr-qc/0006073.pdf , which is the same paper
  // used for the constraint and evolution equations, we get the following lists:
  // Odd: S, K^z_\rho, W, \beta^\rho
  // Even: \Psi, U, \alpha, \beta^z


  //Even Reularity enforcement function

  // Odd regularity should be phased out by implementing a 0 boundary on axis
  // constraint using the affine constraints ability of dealii this may
  // lead to forming of two different affine constraints for even and odd functions
/*
  template <int dim>
  class even_regularity:public Function<dim>
  {
  public:
    template <typename number>
    virtual Tensor<1,dim,VectorizedArray<number>> value(const Point<dim> &p,
                         Tensor<1,dim,VectorizedArray<number>> &beta_vec,
                         const unsigned int component = 0) const override;

    template <typename number>
    Tensor<1, dim, VectorizedArray<number>> value(const Point<dim, VectorizedArray<number>> &p,
                                  Tensor<1,dim,VectorizedArray<number>> &beta_vec,
                                  const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  Tensor<1, dim, VectorizedArray<number>> even_regularity<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 Tensor<1,dim,VectorizedArray<number>> &beta_vec, const unsigned int //component) const
  {
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d){
          if (std::fabs(p[d][v]) < 1e-12)
            beta_vec[d][v] = 0;
        }
      }
    return beta_vec;
  }



  template <int dim>
  template <typename number>
  Tensor<1, dim, VectorizedArray<number>> even_regularity<dim>::value(const Point<dim> & p,
                                 Tensor<1,dim,VectorizedArray<number>> &beta_vec, const unsigned int //component) const
  {
    double rho = abs(p(0));
    double z = p(1);
    if (rho < 1e-12)
      beta_vec[0][0] = 0;
    if (std:fabs(z) < 1e-12)
      beta_vec[1][0] = 0;
    return beta_vec;
  }



  //Odd Regularity Enforcement function

  template <int dim>
  class odd_regularity:public Function<dim>
  {
  public:
    template <typename number>
    virtual double value(const Point<dim> &p,
                         double &function_value,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                                  VectorizedArray<number> &function_value,
                                  const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> odd_regularity<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 VectorizedArray<number> &function_value, const unsigned int //component) const
  {
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d){
          if (std::fabs(p[d][v]) < 1e-12)
            function_value[v] = 0;
        }
      }
    return function_value;
  }

  template <int dim>
  template <typename number>
  double odd_regularity<dim>::value(const Point<dim> & p,
                                 double &function_value, const unsigned int //component) const
  {
    double rho = abs(p(0));
    double z = p(1);
    if (rho < 1e-12)
      function_value = 0.;
    if (std:fabs(z) < 1e-12)
      function_value = 0.;
    return function_value;
  }*/

  template <int dim>
  class const_rho_function_altered:public Function<dim>
  {
  public:

    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> const_rho_function_altered<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d)
          q[d] = p[d][v];
        result[v] = q[0]/(q[0]*q[0]+q[1]*q[1]);
      }

    return result;
  }



  template <int dim>
  double const_rho_function_altered<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    //return value<double>(p, component);
    double rho = p(0);
    double r_sq = rho*rho + p(1)*p(1);
    return rho/r_sq;
  }


  template <int dim>
  class const_z_function_altered:public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> const_z_function_altered<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d)
          q[d] = p[d][v];
        result[v] = q[1]/(q[0]*q[0]+q[1]*q[1]);
      }

    return result;
  }



  template <int dim>
  double const_z_function_altered<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    //return value<double>(p, component);
    // Returns the alpha factor from Robin Boundary Condition (taken from DOI: 10.1103/PhysRevD.91.044033)
    // Extra rho included for integration reasons
    double rho = p(0);
    double z = p(1);
    double r_sq = rho*rho+z*z;
    return z/r_sq;
  }

  template <int dim>
  class initial_drho_S:public Function<dim>
  {
  public:

    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> initial_drho_S<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 const unsigned int /*component*/) const
  {
    double amp =seed_amp;
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d)
          q[d] = p[d][v];
        result[v] = -2.*amp*q[0]*q[0]*exp(-q[0]*q[0]-q[1]*q[1]);
      }

    return result;
  }



  template <int dim>
  double initial_drho_S<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    //return value<double>(p, component);
    double amp =seed_amp;
    double rho = p(0);
    double r_sq = rho*rho + p(1)*p(1);
    return -2.*amp*rho*rho*exp(-r_sq);
  }


  template <int dim>
  class initial_dz_S:public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> initial_dz_S<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        double amp =seed_amp;
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d)
          q[d] = p[d][v];
        result[v] = result[v] = -2.*amp*q[1]*q[0]*exp(-q[0]*q[0]-q[1]*q[1]);
      }

    return result;
  }



  template <int dim>
  double initial_dz_S<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    //return value<double>(p, component);
    // Returns the alpha factor from Robin Boundary Condition (taken from DOI: 10.1103/PhysRevD.91.044033)
    // Extra rho included for integration reasons
    double amp =seed_amp;
    double rho = p(0);
    double z = p(1);
    double r_sq = rho*rho+z*z;
    return -2.*amp*z*rho*exp(-r_sq);
  }


  template <int dim>
  class inv_rho_function_altered:public Function<dim>
  {
  public:

    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> inv_rho_function_altered<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d)
          q[d] = p[d][v];
        result[v] = sqrt(q[0]*q[0]+q[1]*q[1])/q[0];
      }

    return result;
  }



  template <int dim>
  double inv_rho_function_altered<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    //return value<double>(p, component);
    double rho = p(0);
    double r_sq = rho*rho + p(1)*p(1);
    return sqrt(r_sq)/rho;
  }


  template <int dim>
  class inv_z_function_altered:public Function<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> inv_z_function_altered<dim>::value(const Point<dim, VectorizedArray<number>> &p,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> q;
        for (unsigned int d = 0; d < dim; ++d)
          q[d] = p[d][v];
        result[v] = sqrt(q[0]*q[0]+q[1]*q[1])/q[1];
      }

    return result;
  }



  template <int dim>
  double inv_z_function_altered<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    //return value<double>(p, component);
    // Returns the alpha factor from Robin Boundary Condition (taken from DOI: 10.1103/PhysRevD.91.044033)
    // Extra rho included for integration reasons
    double rho = p(0);
    double z = p(1);
    double r_sq = rho*rho+z*z;
    return sqrt(r_sq)/z;
  }

  double factorial(const unsigned int &x){
    double result = 1;
    for (unsigned int i = x; i > 0; --i){
      result = result * i;
    }
    return result;
  }

  template <int dim, typename SystemMatrixType_name, typename LevelMatrixType_name>
  void general_solver(const unsigned int &nlevels,const DoFHandler<dim> &dof_handler,
     AffineConstraints<double> &constrain, MGConstrainedDoFs &mg_constrain,
     SystemMatrixType_name &syst_matrix, MGLevelObject<LevelMatrixType_name> &mg_matrice,
     LinearAlgebra::distributed::Vector<double> &sol, LinearAlgebra::distributed::Vector<double> &rhs)
  {
        Timer                            time;
        /*MGTransferMatrixFree<dim, float> mg_transfer(mg_constrain);
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
                smoother_data[0].eig_cg_n_iterations = mg_matrice[0].m()+2;
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
          preconditioner(dof_handler, mg, mg_transfer);*/

        // The setup of the multigrid routines is quite easy and one cannot see
        // any difference in the solve process as compared to step-16. All the
        // magic is hidden behind the implementation of the LaplaceOperator::vmult
        // operation. Note that we print out the solve time and the accumulated
        // setup time through standard out, i.e., in any case, whereas detailed
        // times for the setup operations are only printed in case the flag for
        // detail_times in the constructor is changed.
        PreconditionIdentity preconditioner_tester;

        SolverControl solver_control(10000, 1e-12 * rhs.l2_norm());
        SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);
        //setup_time += time.wall_time();
        /*time_details << "MG build smoother time     (CPU/wall) " << time.cpu_time()
                     << "s/" << time.wall_time() << "s\n";
        pcout << "Total setup time               (wall) " << setup_time << "s\n";*/

        time.reset();
        time.start();
        //std::cout << "I made it to 1" << std::endl;
        constrain.set_zero(sol);
        //constrain.set_zero(rhs);

        gmres.solve(syst_matrix, sol, rhs, preconditioner_tester);
        //std::cout << "I made it to 3" << std::endl;
        constrain.distribute(sol);

        //std::cout << "Time solve (" << solver_control.last_step() <<
        //" iterations)\n" << std::endl;
  }

  template <typename T>
  int sign_of(T val){
    return (T(0) < val) - (val < T(0));
  }

}//namespace Brill_Evolution
