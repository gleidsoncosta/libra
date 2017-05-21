#include "../include/Data.h"
#include <iostream>
#include <vector>
#include <time.h>

Data::Data()
{
    SetData();
}

void Data::SetData(){

    cin >> rows;
    cin >> cols;
    cin >> classes;
    cin >> func;
    cin >> perc;


    for(int i=0; i<rows; i++){
        vector<double> row;
        for(int j=0; j<cols; j++){
            double val;
            cin >> val;
            row.push_back(val);
        }
        alldata.push_back(row);
    }

    SetPerc();
}

void Data::SetValues(int func, double perc){
    this->func = func;
    this->perc = perc;
}

void Data::SetPerc(){
    int trainingsize = (int)rows*(perc);
    int testsize = rows-trainingsize;
    trainingdata.clear();
    testdata.clear();
    srand(time(NULL));
    vector<vector<double> > aux = alldata;
    vector<vector<double> >::iterator it = aux.begin();
    if(trainingsize>testsize){
        for(int i=0; i<testsize; i++){
            int val = rand() % (aux.size()-1);
            testdata.push_back(aux[val]);
            aux.erase(it+(val));
        }
        trainingdata = aux;
    }else{
        for(int i=0; i<trainingsize; i++){
            int val = rand() % (aux.size()-1);
            trainingdata.push_back(aux[val]);
            aux.erase(it+(val));
        }
        testdata = aux;
    }
}

void Data::printMatrix(vector<vector<double> > d){
    for(int i=0; i<d.size(); i++){
        printArray(d[i]);
        cout<<endl;
    }
}

void Data::printArray(vector<double> d){
    for(int i=0; i<d.size(); i++){
        cout << d[i] << " ";
    }
}
