#ifndef NETWORK_H
#define NETWORK_H
#include "Layer.h"
#include "Neuron.h"
#include <vector>

using namespace std;


class Network
{
    public:
        vector<double> network_layers;
        vector<Layer> layers;

        double alpha;
        int funcao;
        double sqe;

        double epoches=0;
        double iter=0;

        Network(vector<double> net_layers, int funcaoativ, double alpha, double sqe);
        void FeedInput(vector<double> inputs);
        void FowardPropagation();
        void NewFowardPropagation();
        void CalcErrorLayer(double target);
        void AdjustWeights();

        double QuadErrorSum();

        void printNetwork();
        void printResult();
        char* funcaoName();
        double justResult();
        double outp();
};

#endif // NETWORK_H
