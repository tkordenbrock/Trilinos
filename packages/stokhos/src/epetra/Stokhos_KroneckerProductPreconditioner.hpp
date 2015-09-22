// @HEADER
// ***********************************************************************
// 
//                           Stokhos Package
//                 Copyright (2009) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Eric T. Phipps (etphipp@sandia.gov).
// 
// ***********************************************************************
// @HEADER

#ifndef STOKHOS_KRONECKER_PRODUCT_PRECONDITIONER_HPP
#define STOKHOS_KRONECKER_PRODUCT_PRECONDITIONER_HPP

#include "Teuchos_RCP.hpp"

#include "Stokhos_SGPreconditioner.hpp"
#include "EpetraExt_MultiComm.h"
#include "Stokhos_OrthogPolyBasis.hpp"
#include "Stokhos_EpetraSparse3Tensor.hpp"
#include "Epetra_Map.h"
#include "Stokhos_AbstractPreconditionerFactory.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Stokhos {
    
  /*! 
   * \brief An Epetra operator representing applying the mean in a block
   * stochastic Galerkin expansion.
   */
  class KroneckerProductPreconditioner : public Stokhos::SGPreconditioner {
      
  public:

    //! Constructor 
    KroneckerProductPreconditioner(
      const Teuchos::RCP<const EpetraExt::MultiComm>& sg_comm,
      const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& sg_basis,
      const Teuchos::RCP<const Stokhos::EpetraSparse3Tensor>& epetraCijk,
      const Teuchos::RCP<const Epetra_Map>& base_map,
      const Teuchos::RCP<const Epetra_Map>& sg_map,
      const Teuchos::RCP<Stokhos::AbstractPreconditionerFactory>& mean_prec_factory,
      const Teuchos::RCP<Stokhos::AbstractPreconditionerFactory>& G_prec_factory,
      const Teuchos::RCP<Teuchos::ParameterList>& params);
    
    //! Destructor
    virtual ~KroneckerProductPreconditioner();

    /** \name Stokhos::SGPreconditioner methods */
    //@{

    //! Setup preconditioner
    virtual void 
    setupPreconditioner(const Teuchos::RCP<Stokhos::SGOperator>& sg_op, 
			const Epetra_Vector& x);

    //@}

    /** \name Epetra_Operator methods */
    //@{
    
    //! Set to true if the transpose of the operator is requested
    virtual int SetUseTranspose(bool UseTranspose);
    
    /*! 
     * \brief Returns the result of a Epetra_Operator applied to a 
     * Epetra_MultiVector Input in Result as described above.
     */
    virtual int Apply(const Epetra_MultiVector& Input, 
                      Epetra_MultiVector& Result) const;

    /*! 
     * \brief Returns the result of the inverse of the operator applied to a 
     * Epetra_MultiVector Input in Result as described above.
     */
    virtual int ApplyInverse(const Epetra_MultiVector& X, 
                             Epetra_MultiVector& Y) const;
    
    //! Returns an approximate infinity norm of the operator matrix.
    virtual double NormInf() const;
    
    //! Returns a character string describing the operator
    virtual const char* Label () const;
  
    //! Returns the current UseTranspose setting.
    virtual bool UseTranspose() const;
    
    /*! 
     * \brief Returns true if the \e this object can provide an 
     * approximate Inf-norm, false otherwise.
     */
    virtual bool HasNormInf() const;

    /*! 
     * \brief Returns a reference to the Epetra_Comm communicator 
     * associated with this operator.
     */
    virtual const Epetra_Comm & Comm() const;

    /*!
     * \brief Returns the Epetra_Map object associated with the 
     * domain of this matrix operator.
     */
    virtual const Epetra_Map& OperatorDomainMap () const;

    /*! 
     * \brief Returns the Epetra_Map object associated with the 
     * range of this matrix operator.
     */
    virtual const Epetra_Map& OperatorRangeMap () const;

    //@}

  protected:

    //! Compute trace of matrix A'B.
    double MatrixTrace(const Epetra_CrsMatrix& A,
		       const Epetra_CrsMatrix& B) const;

  private:
    
    //! Private to prohibit copying
    KroneckerProductPreconditioner(const KroneckerProductPreconditioner&);
    
    //! Private to prohibit copying
    KroneckerProductPreconditioner& operator=(const KroneckerProductPreconditioner&);
    
  protected:
    
    //! Label for operator
    std::string label;

    //! Stores SG parallel communicator
    Teuchos::RCP<const EpetraExt::MultiComm> sg_comm;

    //! Stochastic Galerking basis
    Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> > sg_basis;

    //! Stores Epetra Cijk tensor
    Teuchos::RCP<const Stokhos::EpetraSparse3Tensor> epetraCijk;
    
    //! Stores base map
    Teuchos::RCP<const Epetra_Map> base_map;

    //! Stores SG map
    Teuchos::RCP<const Epetra_Map> sg_map;

    //! Stores factory for building mean preconditioner
    Teuchos::RCP<Stokhos::AbstractPreconditionerFactory> mean_prec_factory;

    //! Stores factory for building G preconditioner
    Teuchos::RCP<Stokhos::AbstractPreconditionerFactory> G_prec_factory;

    //! Preconditioner parameters
    Teuchos::RCP<Teuchos::ParameterList> params;

    //! Stores mean preconditioner
    Teuchos::RCP<Epetra_Operator> mean_prec;

    //! Stores G preconditioner
    Teuchos::RCP<Epetra_Operator> G_prec;

    //! Flag indicating whether transpose was selected
    bool useTranspose;

    //! Pointer to the SG operator.
    Teuchos::RCP<Stokhos::SGOperator> sg_op;

    //! Pointer to the PCE expansion of Jacobian.
    Teuchos::RCP<Stokhos::EpetraOperatorOrthogPoly > sg_poly;

    //! Short-hand for Cijk
    typedef Stokhos::Sparse3Tensor<int,double> Cijk_type;

    //! Stores triple product tensor
    Teuchos::RCP<const Cijk_type> Cijk;
 
    //! Pointer to CrsMatrix G
    Teuchos::RCP<Epetra_CrsMatrix> G;

    //! Flag indicating whether operator be scaled with <\psi_i^2>
    bool scale_op;

    //! Limit construction of G to linear terms
    bool only_use_linear;

    mutable Teuchos::RCP<Epetra_MultiVector> result_MVT;

  }; // class KroneckerProductPreconditioner
  
} // namespace Stokhos

#endif // STOKHOS_KRONECKER_PRODUCT_PRECONDITIONER_HPP
