#include "../include/Neuron.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <stdlib.h>
#include <time.h>

Neuron::Neuron(int num_con, int funcaoativ, int id)
{
    srand (time(NULL));
    this->id = id;
    num_connections = num_con;
    func = funcaoativ;

    vector<double> b;
    b.push_back(0.8);
    b.push_back(-0.1);
    b.push_back(0.5);
    b.push_back(0.9);
    b.push_back(0.4);
    b.push_back(1.0);
    b.push_back(0.3);
    b.push_back(-1.2);
    b.push_back(1.1);
    /*b.push_back(7.3061);
    b.push_back(2.8441);
    b.push_back(4.7621);
    b.push_back(6.3917);
    b.push_back(4.7618);
    b.push_back(6.3917);
    b.push_back(4.5589);
    b.push_back(10.3788);
    b.push_back(9.7691);*/
    for(int i=0; i<num_connections; i++){
        weights.push_back((double)(rand()%10)/10);
        //weights.push_back(b[id+i]);
    }
    deltaweights.resize(weights.size(), 2);
    error = NULL;
    output = NULL;
}

void Neuron::printNeuron(){
    for(int i=0; i<num_connections; i++){
        cout << "|______w" << id+i << ": " << weights[i] << endl;
        cout << "|_________Dw" << id+i << ": " << deltaweights[i] << endl;
    }
}

void Neuron::FeedFoward(vector<Neuron> prev_layer, int me){

    double sum = Sum(prev_layer, me);
    switch(func){
    case 0:
        output = Sigmoid(sum);
        break;
    case 1:
        output = TanH(sum);
        break;
    case 2:
        output = Gaussian(sum);
        break;
    case 3:
        output = Linear(sum);
        break;
    }
}

double Neuron::Sum(vector<Neuron> prev_layer, int me){

    double sum = 0;
    for(int i=0; i<prev_layer.size(); i++){
        sum += prev_layer[i].weights[me-1] * prev_layer[i].output;
    }
    return sum;
}

double Neuron::Sigmoid(double value){
     return 1 / (1 + exp(-value));
}

double Neuron::TanH(double value){
    return tanh(value);
}

double Neuron::Linear(double value){
    return value;
}

double Neuron::Gaussian(double value){
    return exp(-0.5*pow(value,2));
}

double Neuron::Gradient(int me){
     double grad = this->output*(1-this->output)*this->error;
     return grad;
}

void Neuron::UpdateWeight(){
    for(int i=0; i<num_connections; i++){
        weights[i] = weights[i]+deltaweights[i];
        deltaweights[i] = 0;
    }
}
