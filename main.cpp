#include <iostream>
#include "include/Network.h"
#include "include/bateriasteste.h"
#include "include/TrainingData.h"
#include "include/Data.h"
#include <iomanip>

using namespace std;

Network defineNetwork(vector<double> network_layers, vector<vector<double> > data, vector<double> bateria){

    Network net(network_layers, bateria[0], bateria[1], bateria[2]);
    double last_quad_error = 0;
    //training
    do{
        last_quad_error = net.QuadErrorSum();
        for(int i=0; i<data.size(); i++){
            net.FeedInput(data[i]);
            net.FowardPropagation();
            net.CalcErrorLayer((double)data[i][data[i].size()-1]);
            net.AdjustWeights();
            net.iter++;
        }
        net.epoches++;
        //cout << bateria[0] << " " << bateria[1] << " " << net.epoches << " " << net.iter << " " <<  " " <<  net.QuadErrorSum() << " " << bateria[2] << " " << net.outp() << endl;
    }while(net.QuadErrorSum() > bateria[2] &&  net.epoches < 500000);

    return net;
}

void testNetwork(Network net, vector<vector<double> > data){

    int accuracy = 0;
    //testing
    //cout <<"entrada1;entrada2;saida desejada;funcao ativacao;resultado numerico;resultado aproximado " << endl;
    for(int i=0; i<data.size(); i++){
        net.FeedInput(data[i]);
        net.FowardPropagation();

        double aproximation=-1;
        if(1.0-net.justResult()>=0.5) aproximation = 0.0;
        else aproximation = 1.0;

        if(data[i][2] == aproximation) accuracy++;

    //    cout << data[i][0] << ";" << data[i][1] << ";" << data[i][2] <<";";
    //    cout << net.funcaoName() << ";" << net.justResult() <<";" << aproximation << endl;
    }
    cout << net.epoches<<";"<<net.iter<<";"<<net.funcaoName()<<";"<<net.alpha<<";"<<net.sqe<<";"<<(double)accuracy*100/data.size() << "%" << endl;

}

int main()
{
    Data data;
    Baterias baterias;

    vector<double> network_layers;
    network_layers.push_back(data.cols-1);
    network_layers.push_back(2);
    network_layers.push_back(1);

    cout << "MLP  ";
    for(int i=0; i<network_layers.size(); i++){
        cout << network_layers[i] << " ";
    }
    cout << endl;
    cout << "Epoches;Iteracoes;funcao ativacao;Alpha;Taxa Erro;Acurácia" << endl;

    for(int i=0; i<baterias.testes.size(); i++){
        Network net = defineNetwork(network_layers, data.trainingdata, baterias.testes[i]);
        testNetwork(net, data.alldata);
    }
}
