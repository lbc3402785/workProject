#include "priorcostcallback.h"

PriorCostCallBack::PriorCostCallBack(fitting::PriorCost* cost):cost(cost)
{

}

ceres::CallbackReturnType PriorCostCallBack::operator ()(const ceres::IterationSummary &summary)
{
    if(summary.iteration%5==0)cost->setWeight(cost->getWeight()/2.0);
    if (!summary.step_is_successful) {
      return ceres::SOLVER_CONTINUE;
    }

    return ceres::SOLVER_CONTINUE;
}
