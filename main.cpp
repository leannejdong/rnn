
#include "Layers.h"
#include "Samples.h"
#include "NeuralNetwork.h"
#include "TrainingExamples.h"
#include "Simulation.h"

int main(void) {
    cout << "hello" << endl;

    int VSize = 90;

    ofstream out1;
    out1.open("Oneout1.txt");
    ofstream out2;
    out2.open("Oneout2.txt");
    ofstream out3;
    out3.open("Oneout3.txt");

    for (int hidden = 3; hidden <= 7; hidden += 2) {
        Sizes EvalAverage;
        Layer ErrorAverage;
        Layer CycleAverage;

        Layer NMSETrainAve;
        Layer NMSETestAve;

        int MeanEval = 0;
        double MeanError = 0;
        double MeanCycle = 0;

        double NMSETrainMean = 0;
        double NMSETestMean = 0;

        int EvalSum = 0;

        double NMSETrainSum = 0;
        double NMSETestSum = 0;

        double ErrorSum = 0;
        double CycleSum = 0;
        double maxrun = 30;
        int success = 0;

        double BestRMSE = 3;
        double BestNMSE = 3;

        for (int run = 1; run <= maxrun; run++) {
            Simulation Combined;

            Combined.Procedure(true, hidden, out1, out2, out3);

        } //run

    } //hidden
    out1.close();
    out2.close();
    out3.close();

    return 0;

}
