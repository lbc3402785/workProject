#ifndef PRIORCOSTCALLBACK_H
#define PRIORCOSTCALLBACK_H
#include "ceres/ceres.h"
#include "ceresnonlinear.hpp"

class PriorCostCallBack : public ceres::IterationCallback
{
public:
    PriorCostCallBack(fitting::PriorCost* cost);

    // IterationCallback interface
public:
    ceres::CallbackReturnType operator ()(const ceres::IterationSummary &summary);
private:
    fitting::PriorCost* cost;
};

#endif // PRIORCOSTCALLBACK_H
