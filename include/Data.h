#ifndef DATA_H
#define DATA_H

#include <stdlib.h>
#include <vector>

using namespace std;

class Data
{
    public:
        int rows;
        int cols;
        int func;
        int classes;
        double perc;
        vector<vector<double> > alldata;
        vector<vector<double> > trainingdata;
        vector<vector<double> > testdata;

        Data();
        void SetData();
        void SetValues(int func, double perc);
        void SetPerc();
        void printMatrix(vector<vector<double> > d);
        void printArray(vector<double> d);
};

#endif // DATA_H
