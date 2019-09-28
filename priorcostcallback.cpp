#include "priorcostcallback.h"

PriorCostCallBack::PriorCostCallBack(fitting::PriorCost* cost,float learnRate):cost(cost),learnRate(learnRate)
{

}

ceres::CallbackReturnType PriorCostCallBack::operator ()(const ceres::IterationSummary &summary)
{
    if(summary.iteration%5==0)cost->setWeight(cost->getWeight()*learnRate);
    if (!summary.step_is_successful) {
      return ceres::SOLVER_CONTINUE;
    }

    return ceres::SOLVER_CONTINUE;
}
