#ifndef PRIORCOSTCALLBACK_H
#define PRIORCOSTCALLBACK_H
#include "ceres/ceres.h"
#include "ceresnonlinear.hpp"

class PriorCostCallBack : public ceres::IterationCallback
{
public:
    PriorCostCallBack(fitting::PriorCost* cost,float learnRate=0.8);

    // IterationCallback interface
public:
    ceres::CallbackReturnType operator ()(const ceres::IterationSummary &summary);
private:
    fitting::PriorCost* cost;
    float learnRate;
};

#endif // PRIORCOSTCALLBACK_H
