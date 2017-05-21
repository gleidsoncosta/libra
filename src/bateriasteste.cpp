#include "../include/bateriasteste.h"
#include <iostream>
#include <vector>

Baterias::Baterias()
{
    int rows;
    int cols;
    cin>>rows;
    cin>>cols;

    for(int i=0; i<rows; i++){
        vector<double> row;
        for(int j=0; j<cols; j++){
            double val;
            cin >> val;
            row.push_back(val);
        }
        testes.push_back(row);
    }
}
