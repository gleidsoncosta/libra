#include "../include/Layer.h"
#include <iostream>


Layer::Layer(int layer_def, int next_layer_def, int funcaoativ, int id)
{
    int soma = 0;
    this->id = id;
    layer.push_back(Neuron(next_layer_def, funcaoativ, id+soma));
    layer.back().output = -1;

    for(int i=0; i<layer_def; i++){
        soma+=next_layer_def;
        layer.push_back(Neuron(next_layer_def, funcaoativ, id+soma));
    }
}

void Layer::printLayer(){
    for(int i=0; i<(int)layer.size(); i++){
        cout << "|__Neuronio: " << i << " Valor: " << layer[i].output << " Erro: " << layer[i].error;
        cout << endl;
        layer[i].printNeuron();
    }
}

void Layer::FeedInput(vector<double> input){
    for(int i=1; i<(int)input.size(); i++){
        layer[i].output = input[i-1];
    }
}

void Layer::Foward(Layer prev_layer){

    for(int i=1; i<(int)layer.size(); i++){
        layer[i].FeedFoward(prev_layer.layer, i);
    }
}

void Layer::ErrorOutputLayer(double t){
    for(int i=1; i<(int)layer.size(); i++){
        double y = layer[i].output;
        layer[i].error = (t - y);
    }
}

void Layer::FixWeightsLayer(){
    for(int i=0; i<(int)layer.size(); i++){
        layer[i].UpdateWeight();
    }
}

void Layer::DeltaWeightsLayer(Layer next_layer, double alpha){
    vector<Neuron> nl = next_layer.layer;
    for(int i=0; i<(int)layer.size(); i++){
        for(int j=1; j<(int)nl.size(); j++){
            layer[i].deltaweights[j-1] = alpha*layer[i].output*nl[j].Gradient(j-1);
            layer[i].error = nl[j].Gradient(j-1)*layer[i].weights[j-1];
        }
    }
}
