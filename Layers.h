//
// Created by leanne on 9/7/20.
//

#ifndef RNN_LAYERS_H
#define RNN_LAYERS_H
#include <array>
#include <vector>



using namespace std;

using Data = vector<vector<double> > ;
using Layer = vector<double>;//for the different layers within the neural network
using Weight = vector<vector<double> >;//for managing the weights of the neural network connections-=-a vector matrix of type double

class Layers {
public:
    array<array<double,35>,35> Weights;
    array<array<double,35>,35> WeightChange;
    array<array<double,35>,35> ContextWeight;


    //class variables
    Weight TransitionProb ;//neural network weights

    Data RadialOutput;
    Data Outputlayer;
    Layer Bias;//keeping track of the bias factor in the network
    Layer BiasChange;//keeping track of change in Bias factor
    Data Error;//error after each iteration

    Layer Mean;
    Layer StanDev;

    Layer MeanChange;
    Layer StanDevChange;
};


#endif//RNN_LAYERS_H
