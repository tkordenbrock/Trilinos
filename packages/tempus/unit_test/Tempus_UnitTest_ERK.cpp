// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#include "Teuchos_UnitTestHarness.hpp"

#include "../TestModels/DahlquistTestModel.hpp"

#include "Tempus_StepperFactory.hpp"
#include "Tempus_SolutionHistory.hpp"
#include "Tempus_StepperRKModifierBase.hpp"
#include "Tempus_StepperRKModifierXBase.hpp"


namespace Tempus_Unit_Test {

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_const_cast;
using Teuchos::rcp_dynamic_cast;


/** \brief Explicit RK Modifier Dahlquist Test using StepperERK_BogackiShampine32.
 *
 *  This is the canonical Dahlquist test equation
 *  \f[
 *    \mathbf{\dot{x}} = \lambda \mathbf{x} = \bar{f}(x,t).
 *  \f]
 *  where \f$\lambda = -1\f$ with the initial condition
 *  \f$\mathbf{x}(0) = 1\f$, and the exact solution is
 *  \f[
 *    \mathbf{x}(t) = \exp{[\lambda t]}.
 *  \f]
 *  The general RK method can be written as
 *  \f{eqnarray*}{
 *    X_{i} & = & x_{n-1} +
 *     \Delta t\,\sum_{j=1}^{i-1} a_{ij}\,\bar{f}(X_{j},t_{n-1}+c_{j}\Delta t)\\
 *    x_{n} & = & x_{n-1}
 *    + \Delta t\,\sum_{i=1}^{s}b_{i}\,\bar{f}(X_{i},t_{n-1}+c_{i}\Delta t)
 *  \f}
 *
 *  For the RK Bogacki-Shampine tableau,
 *  \f[
 *    \begin{array}{c|cccc}  0  & 0    &     &     &     \\
 *                          1/2 & 1/2  & 0   &     &     \\
 *                          3/4 & 0    & 3/4 & 0   &     \\
 *                           1  & 2/9  & 1/3 & 4/9 & 0   \\ \hline
 *                              & 2/9  & 1/3 & 4/9 & 0   \\
 *                              & 7/24 & 1/4 & 1/3 & 1/8 \end{array}
 *  \f]
 *  the solution for the first time step is (explicitly writing out the stages)
 *  \f{eqnarray*}{
 *    X_1 & = & x_0 \\
 *    X_2 & = & x_0 + \Delta t\,(1/2)\,\bar{f}(X_1,t_0+( 0 )\Delta t) \\
 *    X_3 & = & x_0 + \Delta t\,(3/4)\,\bar{f}(X_2,t_0+(1/2)\Delta t) \\
 *    X_4 & = & x_0 + \Delta t\,(2/9)\,\bar{f}(X_1,t_0+( 0 )\Delta t)
 *                  + \Delta t\,(1/3)\,\bar{f}(X_2,t_0+(1/2)\Delta t)
 *                  + \Delta t\,(4/9)\,\bar{f}(X_3,t_0+(3/4)\Delta t)
 *  \f}
 *  and
 *  \f{eqnarray*}{
 *    x_1 & = & x_0   + \Delta t\,(2/9)\,\bar{f}(X_1,t_0+( 0 )\Delta t)
 *                    + \Delta t\,(1/3)\,\bar{f}(X_2,t_0+(1/2)\Delta t)
 *                    + \Delta t\,(4/9)\,\bar{f}(X_3,t_0+(3/4)\Delta t) \\
 *    x_1 & = & X_4
 *  \f}
 *  Thus we have
 *  \f{eqnarray*}{
 *                    t_0 & = & 0 \\
 *               \Delta t & = & 1 \\
 *    \mathbf{x}(0) = x_0 & = & 1 \\
 *                  X_{1} & = & x_0 \\
 *                        & = & 1 \\
 *       \bar{f}(X_1, 0 ) & = & \lambda X_1 = (-1)(1) \\
 *                        & = & -1 \\
 *                  X_{2} & = & x_0 + (1/2)\,\bar{f}(X_1, 0 ) = 1 + 1/2(-1) \\
 *                        & = & 1/2 \\
 *       \bar{f}(X_2,1/2) & = & \lambda X_2 = (-1)(1/2) \\
 *                        & = & -1/2 \\
 *                    X_3 & = & x_0 + (3/4)\,\bar{f}(X_2,1/2) = 1 + (3/4)(-1/2)\\
 *                        & = & 5/8 \\
 *       \bar{f}(X_3,3/4) & = & \lambda X_3 = (-1)(5/8) \\
 *                        & = & -5/8 \\
 *                    X_4 & = & x_0 + (2/9)\,\bar{f}(X_1,0)
 *                                  + (1/3)\,\bar{f}(X_2,1/2)
 *                                  + (4/9)\,\bar{f}(X_3,3/4) \\
 *                        & = & 1 + (2/9)(-1) + (1/3)(-1/2) + (4/9)(-5/8) \\
 *                        & = & 1/3 \\
 *                    x_1 & = & X_4 = 1/3
 *  \f}
 */
class StepperRKModifierBogackiShampineTest
  : virtual public Tempus::StepperRKModifierBase<double>
{
public:

  /// Constructor
  StepperRKModifierBogackiShampineTest(Teuchos::FancyOStream &Out, bool &Success)
    : out(Out), success(Success)
  {}

  /// Destructor
  virtual ~StepperRKModifierBogackiShampineTest(){}

  /// Test the modify RK Stepper at the Action Locations.
  virtual void modify(
    Teuchos::RCP<Tempus::SolutionHistory<double> > sh,
    Teuchos::RCP<Tempus::StepperRKBase<double> > stepper,
    const typename Tempus::StepperRKAppAction<double>::ACTION_LOCATION actLoc)
  {
    auto stageNumber = stepper->getStageNumber();
    Teuchos::SerialDenseVector<int,double> c = stepper->getTableau()->c();

    auto currentState   = sh->getCurrentState();
    auto workingState   = sh->getWorkingState();
    const double timeCS = currentState->getTime();
    const double dt     = workingState->getTimeStep();
    double       time   = timeCS + c(stageNumber)*dt;
    auto x              = workingState->getX();
    auto xDot = sh->getWorkingState()->getXDot();
    if (xDot == Teuchos::null) xDot = stepper->getStepperXDot();


    if (actLoc == StepperRKAppAction<double>::BEGIN_STEP) {
      const double x_0    = get_ele(*(x), 0);
      const double xDot_0 = get_ele(*(xDot), 0);
      time = timeCS;
      TEST_FLOATING_EQUALITY(x_0,     1.0, 1.0e-15);      // Should be x_0
      TEST_FLOATING_EQUALITY(xDot_0, -1.0, 1.0e-15);      // Should be xDot_0
      TEST_FLOATING_EQUALITY(time,    0.0, 1.0e-15);
      TEST_FLOATING_EQUALITY(dt,      1.0, 1.0e-15);
      TEST_COMPARE(stageNumber, ==, -1);

    } else if (actLoc == StepperRKAppAction<double>::BEGIN_STAGE          ||
               actLoc == StepperRKAppAction<double>::BEFORE_SOLVE         ||
               actLoc == StepperRKAppAction<double>::AFTER_SOLVE          ||
               actLoc == StepperRKAppAction<double>::BEFORE_EXPLICIT_EVAL ||
               actLoc == StepperRKAppAction<double>::END_STAGE) {
      const double X_i = get_ele(*(x), 0);
      const double f_i = get_ele(*(xDot), 0);
      if (stageNumber == 0) {
        TEST_FLOATING_EQUALITY(X_i,      1.0, 1.0e-15);   // Should be X_1
        TEST_FLOATING_EQUALITY(time,     0.0, 1.0e-15);
        if (actLoc == StepperRKAppAction<double>::END_STAGE) {
          TEST_FLOATING_EQUALITY(f_i,     -1.0, 1.0e-15); // Should be \bar{f}_1
        } else {
          TEST_FLOATING_EQUALITY(f_i,      0.0, 1.0e-15); // Not set yet.
        }

      } else if (stageNumber == 1) {
        TEST_FLOATING_EQUALITY(X_i,      0.5, 1.0e-15);   // Should be X_2
        TEST_FLOATING_EQUALITY(time,     0.5, 1.0e-15);
        if (actLoc == StepperRKAppAction<double>::END_STAGE) {
          TEST_FLOATING_EQUALITY(f_i,     -0.5, 1.0e-15); // Should be \bar{f}_2
        } else {
          TEST_FLOATING_EQUALITY(f_i,      0.0, 1.0e-15); // Not set yet.
        }

      } else if (stageNumber == 2) {
        TEST_FLOATING_EQUALITY(X_i,  5.0/8.0, 1.0e-15);   // Should be X_3
        TEST_FLOATING_EQUALITY(time,    0.75, 1.0e-15);
        if (actLoc == StepperRKAppAction<double>::END_STAGE) {
          TEST_FLOATING_EQUALITY(f_i, -5.0/8.0, 1.0e-15); // Should be \bar{f}_3
        } else {
          TEST_FLOATING_EQUALITY(f_i,      0.0, 1.0e-15); // Not set yet.
        }

      } else if (stageNumber == 3) {
        TEST_FLOATING_EQUALITY(X_i,  1.0/3.0, 1.0e-15);   // Should be X_4
        TEST_FLOATING_EQUALITY(time,     1.0, 1.0e-15);
        if (actLoc == StepperRKAppAction<double>::END_STAGE) {
          TEST_FLOATING_EQUALITY(f_i, -1.0/3.0, 1.0e-15); // Should be \bar{f}_4
        } else {
          TEST_FLOATING_EQUALITY(f_i,      0.0, 1.0e-15); // Not set yet.
        }

    } else {
      TEUCHOS_TEST_FOR_EXCEPT( !(-1 < stageNumber && stageNumber < 4));
    }
    TEST_FLOATING_EQUALITY(dt,   1.0, 1.0e-15);

  } else if (actLoc == StepperRKAppAction<double>::END_STEP) {
    const double x_1 = get_ele(*(x), 0);
    time = workingState->getTime();
    TEST_FLOATING_EQUALITY(x_1,  1.0/3.0, 1.0e-15);   // Should be x_1
    TEST_FLOATING_EQUALITY(time,     1.0, 1.0e-15);
    TEST_FLOATING_EQUALITY(dt,       1.0, 1.0e-15);
    TEST_COMPARE(stageNumber, ==, -1);

    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error - unknown action location.\n");
    }
  }

private:

  Teuchos::FancyOStream & out;
  bool & success;
};


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(ERK, Modifier)
{
  auto stepper = rcp(new Tempus::StepperERK_BogackiShampine32<double>());
  Teuchos::RCP<const Thyra::ModelEvaluator<double> >
    model = rcp(new Tempus_Test::DahlquistTestModel<double>());
  auto modifier = rcp(new StepperRKModifierBogackiShampineTest(out, success));

  stepper->setModel(model);
  stepper->setAppAction(modifier);
  stepper->setICConsistency("Consistent");
  stepper->setUseFSAL(true);
  stepper->initialize();

  // Create a SolutionHistory.
  auto solutionHistory = Tempus::createSolutionHistoryME(model);

  // Take one time step.
  stepper->setInitialConditions(solutionHistory);
  solutionHistory->initWorkingState();
  double dt = 1.0;
  solutionHistory->getWorkingState()->setTimeStep(dt);
  solutionHistory->getWorkingState()->setTime(dt);
  stepper->takeStep(solutionHistory);               // Primary testing occurs in
                                                    // modifierX during takeStep().
  // Test stepper properties.
  TEUCHOS_ASSERT(stepper->getOrder() == 3);
}




/** \brief Explicit RK ModifierX Dahlquist Test using StepperERK_BogackiShampine32.
 *
 *  This is similar to StepperRKModifierBogackiShampineTest but is
 *  specialized for the AppAction::ModifierX interface.  See
 *  StepperRKModifierXBogackiShampineTest for more details.
 */
class StepperRKModifierXBogackiShampineTest
  : virtual public Tempus::StepperRKModifierXBase<double>
{
public:

  /// Constructor
  StepperRKModifierXBogackiShampineTest(Teuchos::FancyOStream &Out, bool &Success)
    : out(Out), success(Success)
  {}

  /// Destructor
  virtual ~StepperRKModifierXBogackiShampineTest(){}

  /// Test the modify RK Stepper at the Action Locations.
  virtual void modify(
    Teuchos::RCP<Thyra::VectorBase<double> > x,
    const double time, const double dt, const int stageNumber,
    const typename Tempus::StepperRKModifierXBase<double>::MODIFIER_TYPE modType)
  {
    if (modType == StepperRKModifierXBase<double>::X_BEGIN_STEP) {
      const double x_0 = get_ele(*(x), 0);               // Should be x_0
      TEST_FLOATING_EQUALITY(x_0,  1.0, 1.0e-15);
      TEST_FLOATING_EQUALITY(time, 0.0, 1.0e-15);
      TEST_FLOATING_EQUALITY(dt,   1.0, 1.0e-15);
      TEST_COMPARE(stageNumber, ==, -1);

    } else if (modType == StepperRKModifierXBase<double>::X_BEGIN_STAGE          ||
               modType == StepperRKModifierXBase<double>::X_BEFORE_SOLVE         ||
               modType == StepperRKModifierXBase<double>::X_AFTER_SOLVE          ||
               modType == StepperRKModifierXBase<double>::X_BEFORE_EXPLICIT_EVAL ||
               modType == StepperRKModifierXBase<double>::X_END_STAGE) {
      const double X_i = get_ele(*(x), 0);
      if (stageNumber == 0) {
        TEST_FLOATING_EQUALITY(X_i,      1.0, 1.0e-15);  // Should be X_1
        TEST_FLOATING_EQUALITY(time,     0.0, 1.0e-15);
      } else if (stageNumber == 1) {
        TEST_FLOATING_EQUALITY(X_i,      0.5, 1.0e-15);  // Should be X_2
        TEST_FLOATING_EQUALITY(time,     0.5, 1.0e-15);
      } else if (stageNumber == 2) {
        TEST_FLOATING_EQUALITY(X_i,  5.0/8.0, 1.0e-15);  // Should be X_3
        TEST_FLOATING_EQUALITY(time,    0.75, 1.0e-15);
      } else if (stageNumber == 3) {
        TEST_FLOATING_EQUALITY(X_i,  1.0/3.0, 1.0e-15);  // Should be X_4
        TEST_FLOATING_EQUALITY(time,     1.0, 1.0e-15);
      } else {
        TEUCHOS_TEST_FOR_EXCEPT( !(-1 < stageNumber && stageNumber < 4));
      }
      TEST_FLOATING_EQUALITY(dt,   1.0, 1.0e-15);

    } else if (modType == StepperRKModifierXBase<double>::X_END_STEP) {
      const double x_1 = get_ele(*(x), 0);
      TEST_FLOATING_EQUALITY(x_1,  1.0/3.0, 1.0e-15);   // Should be x_1
      TEST_FLOATING_EQUALITY(time,     1.0, 1.0e-15);
      TEST_FLOATING_EQUALITY(dt,       1.0, 1.0e-15);
      TEST_COMPARE(stageNumber, ==, -1);

    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Error - unknown action location.\n");
    }
  }

private:

  Teuchos::FancyOStream & out;
  bool & success;
};


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(ERK, ModifierX)
{
  auto stepper = rcp(new Tempus::StepperERK_BogackiShampine32<double>());
  Teuchos::RCP<const Thyra::ModelEvaluator<double> >
    model = rcp(new Tempus_Test::DahlquistTestModel<double>());
  auto modifierX = rcp(new StepperRKModifierXBogackiShampineTest(out, success));

  stepper->setModel(model);
  stepper->setAppAction(modifierX);
  stepper->setICConsistency("Consistent");
  stepper->setUseFSAL(true);
  stepper->initialize();

  // Create a SolutionHistory.
  auto solutionHistory = Tempus::createSolutionHistoryME(model);

  // Take one time step.
  stepper->setInitialConditions(solutionHistory);
  solutionHistory->initWorkingState();
  double dt = 1.0;
  solutionHistory->getWorkingState()->setTimeStep(dt);
  solutionHistory->getWorkingState()->setTime(dt);
  stepper->takeStep(solutionHistory);               // Primary testing occurs in
                                                    // modifierX during takeStep().
  // Test stepper properties.
  TEUCHOS_ASSERT(stepper->getOrder() == 3);
}


} // namespace Tempus_Test
