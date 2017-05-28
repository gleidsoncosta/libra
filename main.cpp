#include <iostream>
#include "include/Network.h"
#include "include/bateriasteste.h"
#include "include/Data.h"
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

void ImageProcessing(){
   // http://www.learnopencv.com/histogram-of-oriented-gradients/
    //string filepath = "/home/gleidson/Documentos/Monografia/Aquisicao/dia1/150517101214/0.png";
    string filepath = "/home/gleidson/Documentos/Monografia/MonoQT/BackCurvature/Aquisicao/dia1/150517100604/3.png";

    Mat original;
    original =  imread(filepath, CV_LOAD_IMAGE_COLOR);;
    if( original.data == NULL){
        cout <<  "Could not open or find the image" << endl ;
        return;
    }


    original.convertTo(original, CV_32F, 1/255.0);

    // Calculate gradients gx, gy
    Mat gx, gy;
    Sobel(original, gx, CV_32F, 1, 0, 1);
    Sobel(original, gy, CV_32F, 0, 1, 1);

    Mat mag, angle, magconv;
    cartToPolar(gx, gy, mag, angle, 1);

    mag.convertTo(magconv, CV_8U, 255, 0);

    Mat maggray, edge, dst;
    dst.create(magconv.size(), magconv.type());
    cvtColor(magconv, maggray, CV_BGR2GRAY);

    int edgeThresh = 1;
    int lowThreshold = 5;
    int const max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;

    /// Reduce noise with a kernel 3x3
    //blur( maggray, edge, Size(3,3) );
    edge = maggray;
    //dilates(edge, edge);erosion(edge, edge);

    for(int i=0; i<edge.rows; i++){
        for(int j=0; j<edge.cols; j++){
            if(edge.at<uchar>(i,j)>20)   edge.at<uchar>(i,j)=255;
            else edge.at<uchar>(i,j) = 0;
        }
    }
    imshow("edge", edge);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int largest_area = 0;
    int largest_contour_index = 0;

    /// Canny detector
    //Canny( edge, edge, lowThreshold, lowThreshold*ratio, kernel_size );
    /// Find contours
    findContours( edge, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    for (int i = 0; i< (int)contours.size(); i++) // iterate through each contour.
    {
        double a = contourArea(contours[i], false);  //  Find the area of contour
        if (a>largest_area)
        {
            largest_area = a;
            largest_contour_index = i;                //Store the index of largest contour
        }
    }

    /// Draw contours
    Mat drawing = Mat::zeros( edge.size(), CV_8UC3 );
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( drawing, contours, largest_contour_index, color, 2, 8, hierarchy, 0, Point() );


    imshow("contorno", drawing);
    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    maggray.copyTo( dst, edge);
    imshow("orig", maggray);
    imshow("teste", dst );

    waitKey(0);
    return;                                          // Wait for a keystroke in the window
}

int main()
{
    //Neural();
    ImageProcessing();
}
