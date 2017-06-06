#include <iostream>
#include "include/Network.h"
#include "include/bateriasteste.h"
#include "include/Data.h"
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <dirent.h>
#include <math.h>

using namespace std;
using namespace cv;


RNG rng(12345);

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

void Neural(){
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

void erosion(Mat &resp, Mat frame){
    int dilation_size = 2;
    Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
    erode(frame, resp, element );
}

void dilates(Mat &resp, Mat frame){
    int dilation_size = 2;
    Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
    dilate( frame, resp, element );
}

vector<string> getDirContent(string folder){
    vector<string> cont(0);

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (folder.c_str())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
        cont.push_back(ent->d_name);
      }
      closedir (dir);
    }

    return cont;
}

vector<float> hogDesc(string path){
    vector<float> caract;

    Mat original;
    original =  imread(path.c_str(), CV_LOAD_IMAGE_COLOR);;
    if( original.data == NULL){
        cout <<  "Could not open or find the image" << endl ;
        return caract;
    }

    HOGDescriptor hog( Size(64,64), Size(16,16), Size(8,8), Size(8,8), 9);

    resize(original,original,Size(64,64));

    hog.compute( original, caract);

    waitKey(5);
    return caract;
}

void ImageProcessing(){
    string folderpath = "/home/gleidson/Documentos/NeuralNetwork/mlp/Dataset/Fold1/";
    vector<vector<float> > imgs_features;
    vector<string> imgs_labels;
    //para cada pasta
        //para cada imagem
            //obtem-se as características
                //se foi possível obter-las, salva as caracteríticas e o rótulo (nome da pasta com as imagens)
    vector<string> folders = getDirContent(folderpath);
    for(int i=0; i<folders.size(); i++){
        string filepath = folderpath+folders[i]+"/";
        vector<string> files = getDirContent(filepath);
        for(int j=0; j<files.size(); j++){
            if (strstr(files[j].c_str(),"c")){
                string fullfilepath = filepath+files[j];
                vector<float> row = hogDesc(fullfilepath);
                if(!row.empty()){
                    imgs_features.push_back(row);
                    imgs_labels.push_back(folders[i]);
                }
            }
        }
    }

    cout << imgs_features.size() << endl;
    cout << imgs_labels.size() << endl;

}

int main()
{
    //Neural();
    ImageProcessing();
}
