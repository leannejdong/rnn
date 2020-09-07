//
// Created by leanne on 9/7/20.
//

#ifndef RNN_SIMULATION_H
#define RNN_SIMULATION_H
#include "TrainingExamples.h"
#include "NeuralNetwork.h"
const double learningRate = 0.2;
const double MinimumError = 0.00001;

const int trainsize = 299;
const int testsize = 99;

string trainfile = "train_embed.txt";
string testfile = "test_embed.txt";

class Simulation {

public:
    int TotalEval;
    int TotalSize;
    double Train;
    double Test;
    double TrainNMSE;
    double TestNMSE;
    double Error;

    int Cycles;
    bool Sucess;

    Simulation() {

    }

    int GetEval() {
        return TotalEval;
    }
    double GetCycle() {
        return Train;
    }
    double GetError() {
        return Test;
    }

    double NMSETrain() {
        return TrainNMSE;
    }
    double NMSETest() {
        return TestNMSE;
    }

    bool GetSucess() {
        return Sucess;
    }

    void Procedure(bool bp, double h, ofstream &out1, ofstream &out2,
                   ofstream &out3);
};

void Simulation::Procedure(bool bp, double h, ofstream &out1, ofstream &out2,
                           ofstream &out3) {

    clock_t start = clock();

    int hidden = h;

    int output = 1;
    int input = 1;

    int weightsize1 = (input * hidden);

    int weightsize2 = (hidden * output);

    int contextsize = hidden * hidden;
    int biasize = hidden + output;

    int gene = 1;
    double trainpercent = 0;
    double testpercent = 0;
    int epoch;
    double testtree;

    char file[15] = "Learnt.txt";
    TotalEval = 0;
    double H = 0;

    TrainingExamples Samples(trainfile, trainsize, input, output);
    // Samples.printData();

    double error;

    Sizes layersize;
    layersize.push_back(input);
    layersize.push_back(hidden);
    layersize.push_back(output);

    NeuralNetwork network(layersize);
    network.CreateNetwork(layersize, trainsize);

    epoch = network.BackPropogation(Samples, learningRate, layersize, file,
                                    trainfile, trainsize, input, output, out1, out2); //  train the network

    out2 << "Train" << endl;
    Train = network.TestTrainingData(layersize, file, trainfile, trainsize,
                                     input, output, out2);
    TrainNMSE = network.NMSError();
    out2 << "Test" << endl;
    Test = network.TestTrainingData(layersize, file, testfile, testsize, input,
                                    output, out2);
    TestNMSE = network.NMSError();
    out2 << endl;
    cout << Test << " was test RMSE " << endl;
    out1 << endl;
    out1 << " ------------------------------ " << h << "  " << TotalEval
         << "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE
         << " " << TestNMSE << endl;

    out2 << " ------------------------------ " << h << "  " << TotalEval << "  "
         << Train << "  " << Test << endl;
    out3 << "  " << h << "  " << TotalEval << "  RMSE:  " << Train << "  "
         << Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;

};

#endif//RNN_SIMULATION_H
