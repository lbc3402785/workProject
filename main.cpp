#include "test.h"

int main(int argc, char *argv[])
{
//    testMul();
    if(argc<3){
        std::cout<<"useage:exe modelPath imagePath";
        exit(EXIT_FAILURE);
    }
    std::string modelPath=argv[1];
    std::string imagePath=argv[2];
    Test::testCeres(modelPath,imagePath);
    return 0;
}
