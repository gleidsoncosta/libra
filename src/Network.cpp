#include "../include/Network.h"
#include <iostream>

#include <math.h>
Network::Network(vector<double> net_layers, int funcaoativ, double alpha, double sqe)
{
    this->alpha = alpha;
    this->funcao = funcaoativ;
    this->sqe = sqe;

    int soma = 0;
    network_layers = net_layers;
    for(int i=0; i<network_layers.size(); i++){
        if(i == network_layers.size()-1)    layers.push_back(Layer(network_layers[i], 0, funcaoativ, i+soma));
        else{
            if(i==0)    layers.push_back(Layer(network_layers[i], network_layers[i+1], funcaoativ, i+soma));
            else layers.push_back(Layer(network_layers[i], network_layers[i+1], funcaoativ, i+soma));
            soma+=network_layers[i]+network_layers[i+1]+1;
        }
    }
}

void Network::printNetwork(){
    if(network_layers.size()<3) return;
    for(int i=0; i<layers.size(); i++){
        cout << "Camada: " << i << endl;
        layers[i].printLayer();
        cout << endl;
    }
}

void Network::printResult(){
    if(network_layers.size()<3) return;
    for(int i=1; i<layers[layers.size()-1].layer.size(); i++){
        cout << "Valor "<< i << ": " << layers[layers.size()-1].layer[i].output << endl;
    }
    cout << endl;
}

char* Network::funcaoName(){
    char* nome;
    switch(funcao){
    case 0:
        nome = "Sigmoid";
        break;
    case 1:
        nome = "TanH";
        break;
    case 2:
        nome = "Gaussian";
        break;
    case 3:
        nome = "Linear";
        break;
    }
    return nome;
}

double Network::justResult(){
    return layers[layers.size()-1].layer[1].output;
}

void Network::FeedInput( vector<double> inputs){
    layers.front().FeedInput(inputs);
}

void Network::FowardPropagation(){
    for(int i=1; i<layers.size(); i++){
        layers[i].Foward(layers[i-1]);
    }
}

void Network::CalcErrorLayer(double target){
    layers[layers.size()-1].ErrorOutputLayer(target);
    for(int i=layers.size()-1; i>0; i--){
        layers[i-1].DeltaWeightsLayer(layers[i], alpha);
    }
}

void Network::AdjustWeights(){
    for(int i=0; i<(int)layers.size(); i++){
        layers[i].FixWeightsLayer();
    }
}

double Network::outp(){
    return layers[layers.size()-1].layer[1].output;
}

double Network::QuadErrorSum(){
    double sum = 0;
    for(int i=1; i<layers[layers.size()-1].layer.size(); i++){
        sum+= (double)pow(0.5*layers[layers.size()-1].layer[i].error,2);
    }
    return sum;
}
