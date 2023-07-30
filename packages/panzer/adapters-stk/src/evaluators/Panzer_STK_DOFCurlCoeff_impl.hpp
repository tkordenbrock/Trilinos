// @HEADER
// ***********************************************************************
//
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Roger P. Pawlowski (rppawlo@sandia.gov) and
// Eric C. Cyr (eccyr@sandia.gov)
// ***********************************************************************
// @HEADER

#ifndef __Panzer_DOF_CurlCoeff_impl_hpp__
#define __Panzer_DOF_CurlCoeff_impl_hpp__

#include <cmath>

#include "Phalanx_DataLayout_MDALayout.hpp"

#include "Panzer_STK_Interface.hpp"
#include "Panzer_Workset.hpp"

namespace panzer {

//**********************************************************************
template <typename EvalT,typename Traits>
DOF_CurlCoeff<EvalT,Traits>::DOF_CurlCoeff(const std::string & name,
                                           const Teuchos::RCP<PHX::DataLayout> & data_layout,
                                           const Teuchos::RCP<panzer_stk::STK_Interface> & mesh,
                                           Teuchos::RCP<const panzer::PureBasis> basis)
  : fieldName_(name), mesh_(mesh), basis_(basis)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  dummyField = rcp(new PHX::Tag<ScalarT>("DOF_CurlCoeff: " + name,rcp(new PHX::MDALayout<panzer::Dummy>(0))));
  this->addEvaluatedField(*dummyField);

  source = PHX::MDField<const ScalarT,Cell,Point>(name, data_layout);
  this->addDependentField(source);
  
  std::string n = "DOF_CurlCoeff";
  this->setName(n);
}

template <typename EvalT,typename Traits>
int DOF_CurlCoeff<EvalT,Traits>::cardinalityDigits(Teuchos::RCP<const panzer::PureBasis> &pb)
{
  int n=pb->cardinality();
  int digits=0;
  while (n != 0) {
    n = n / 10;
    ++digits;
  }
  return digits;
}
template <typename EvalT,typename Traits>
std::string DOF_CurlCoeff<EvalT,Traits>::getVariableName(Teuchos::RCP<const panzer::PureBasis> &pb,
                                                         const std::string &name,
                                                         int dof_ordinal)
{
  std::stringstream ss;
  ss << name << "_";
  ss << std::setfill('0') << std::setw(cardinalityDigits(pb));
  ss << dof_ordinal+1;
  return ss.str();
}

//**********************************************************************
template <typename EvalT,typename Traits>
void DOF_CurlCoeff<EvalT,Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  // for convenience pull out some objects from workset
  const std::vector<std::size_t> & localCellIds = this->wda(workset).cell_local_ids;
  std::string blockId = this->wda(workset).block_id;

  auto cellValue_v = Kokkos::View<double*>("cell dof view", source.extent(0));
  auto source_v = source.get_static_view();

  std::cout << "DOF_CurlCoeff: Name = \"" << source.fieldTag().identifier() << "\" at t = " << workset.time << "\n";
  for (int point = 0; point < source.extent_int(1); ++point) {
    Kokkos::parallel_for(source_v.extent(0), KOKKOS_LAMBDA (int i) {
        cellValue_v(i) = Sacado::scalarValue(source_v(i,point));
      });
    Kokkos::fence();
    std::string varName = getVariableName(basis_, fieldName_, point);
    varName = "EDGE_COEFF_" + varName;
    mesh_->setCellFieldData(varName,blockId,localCellIds,cellValue_v);
  }
}

//**********************************************************************
}

#endif
