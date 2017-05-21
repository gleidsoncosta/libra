#ifndef NEURON_H
#define NEURON_H
#include <vector>

using namespace std;


class Neuron
{
    public:
        int id;
        int num_connections;
        int func;

        vector<double> weights;
        vector<double> deltaweights;
        double output;
        double error;

        Neuron(int num_con, int funcaoativ, int id);
        void FeedFoward(vector<Neuron> prev_layer, int me);
        double Sum(vector<Neuron> prev_layer, int me);
        double Sigmoid(double value);
        double TanH(double value);
        double Gaussian(double value);
        double Linear(double value);

        double Gradient(int me);
        void UpdateWeight();

        void printNeuron();
};

#endif // NEURON_H
