#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include "Neuron.h"

#define ALPHA 0.001

using namespace std;


class Layer
{
    public:
        vector<Neuron> layer;

        int id =0;
        Layer(int layer_def, int next_layer_def, int funcaoativ, int id);
        void FeedInput(vector<double> input);
        void Foward(Layer prev_layer);
        void ErrorOutputLayer(double t);
        void FixWeightsLayer();
        void DeltaWeightsLayer(Layer prev_layer, double alpha);
        double CalcDotProductWeight(Layer prev_layer, int neuron);

        void printLayer();
};

#endif // LAYER_H
