//
// Created by leanne on 9/6/20.
//
#ifndef RNN_SAMPLES_H
#define RNN_SAMPLES_H
#include<vector>

using std::vector;
using Layer = vector<double>;//for the different layers within the neural network
using Data = vector<vector<double> > ;

class Samples {
public:

    Data InputValues;
    Data DataSet;
    Layer OutputValues;
    int PhoneSize;
public:
    Samples() {
    }

};



#endif//RNN_SAMPLES_H
