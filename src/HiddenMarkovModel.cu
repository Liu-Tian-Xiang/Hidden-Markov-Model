
#include "HiddenMarkovModel.hpp"

using HMM::ModelSpace;

int main(int argc, char* argv[])
{

    int iterations = atoi(argv[1]);
    ModelSpace o;
    return o.learn(iterations);
}


