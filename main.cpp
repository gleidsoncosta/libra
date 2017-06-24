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

#include <sstream>

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

vector<float> hogDesc(Mat original){
    vector<float> caract;

    resize(original,original,Size(64,64));
                        //w                         //s         //c
    HOGDescriptor hog( Size(64,64), Size(16,16), Size(16,16), Size(16,16), 9);

    hog.compute( original, caract);
    waitKey(5);
    return caract;
}

vector<float> shapeDesc(Mat original){
    /**http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=convex
       procura por 'convexHull' e 'convexityDefects' pra ter uma base das funçoes
       deu suporte na criação do algoritmo:
       https://stackoverflow.com/questions/18401438/find-point-of-convexity-defects-in-opencv-c-function**/


    vector<float> caract;
    Mat gray;
    cvtColor( original, gray, CV_BGR2GRAY );

    //mesmo as binarias, não estao binarias certinho. Tem uns ruidos louco. Ai
    //esse loop é so pra forçar a binarizacao

    for(int i=0; i<gray.rows; i++){
        for(int j=0; j<gray.cols; j++){
            if(gray.at<uchar>(i,j) >= 128)  gray.at<uchar>(i,j) = 255;
            else gray.at<uchar>(i,j) = 0;
        }
     }

    vector<vector<Point> > contours;
    //funcao que encontra os contornos
    findContours(gray, contours, RETR_TREE, CHAIN_APPROX_NONE);

    //descritores -> hull indexes para os pontos extremos
    // -> convDef para os indexes de defeito
    // hullpts e defectpts sao os pontos registrados nos indexes
    vector<vector<int> >hull( contours.size() );
    vector<vector<Vec4i> > convDef(contours.size() );
    vector<vector<Point> > hullpts(contours.size());
    vector<vector<Point> > defectpts(contours.size());

    int numhullpts = 0;
    int numdefpts = 0;


    for(size_t i = 0; i < contours.size(); i++)
    {
        size_t count = contours[i].size();
        //somente um limiar, para caso o numero de pontos de um contorno encontrado
        //não seja possvel criar uma mao. saca?
        if( count < 10 )
            continue;

        //encontra os indices de pontos de extremo
        convexHull( contours[i], hull[i], false );
        numhullpts = hull.size();
        //encontra os indices de pontos de convexidade
        convexityDefects(contours[i], hull[i], convDef[i]);
        for(int k=0;k<convDef[i].size();k++){
            if(convDef[i][k][3]>500){
                numdefpts ++;
            }
        }

        /**DAQUI PRA BAIXO UTILIZA CASO QUEIRA VISUALIZAR OS PONTOS ENCONTRADOS NA IMAGEM,
        POREM COMO DADOS O NUMERO DE PONTOS EXTREMOS E DE DEFEITOS CREIO Q POR ENQUNTO SEJA MAIS IMPORTANTE**/

        /**
        //salva os pontos baseados nos indices encontrados dos pontos de extremo
        for(int k=0;k<hull[i].size();k++){
            int ind=hull[i][k];
            hullpts[i].push_back(contours[i][ind]);
        }

        //encontra os pontos de defeito que satisfaça a condição da distância da reta para com a mão
        for(int k=0;k<convDef[i].size();k++){
            if(convDef[i][k][3]>500){
                int ind_0=convDef[i][k][0];
                int ind_1=convDef[i][k][1];
                int ind_2=convDef[i][k][2];
                cout << contours[i][ind_2] << endl;

                defectpts[i].push_back(contours[i][ind_2]);
                circle(original,contours[i][ind_0],2,Scalar(0,255,0),1);
                circle(original,contours[i][ind_1],2,Scalar(0,255,0),1);
                circle(original,contours[i][ind_2],2,Scalar(255,0,0),1);
                line(original,contours[i][ind_2],contours[i][ind_0],Scalar(0,0,255),1);
                line(original,contours[i][ind_2],contours[i][ind_1],Scalar(0,0,255),1);
            }
        }

        for( int i = 0; i< contours.size(); i++ ){
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( original, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
            drawContours( original, hullpts, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        }
        **/
    }

    caract.push_back((float)numhullpts);
    caract.push_back((float)numdefpts);

    /**imshow("result", original);**/

    //waitKey(0);
    return caract;
}

int partOfGroup(string val, vector<int> groupsidx = vector<int>(0)){
    vector< vector<string> > groups(12);

    groups[0] = vector<string>(4);
    groups[0][0] = "A";groups[0][1] = "E";groups[0][2] = "S";groups[0][3] = "I";

    groups[1] = vector<string>(5);
    groups[1][0] = "U";groups[1][1] = "V";groups[1][2] = "R";groups[1][3] = "W";groups[1][4] = "4";

    groups[2] = vector<string>(3);
    groups[2][0] = "M";groups[2][1] = "N";groups[2][2] = "Q";

    groups[3] = vector<string>(2);
    groups[3][0] = "Adulto";groups[3][1] = "B";

    groups[4] = vector<string>(3);
    groups[4][0] = "C";groups[4][1] = "Pequeno";groups[4][2] = "O";

    groups[5] = vector<string>(2);
    groups[5][0] = "F";groups[5][1] = "T";

    groups[6] = vector<string>(4);
    groups[6][0] = "X";groups[6][1] = "5";groups[6][2] = "7";groups[6][3] = "P";

    groups[7] = vector<string>(5);
    groups[7][0] = "L";groups[7][1] = "G";groups[7][2] = "Palavra";groups[7][3] = "Y";groups[7][4] = "Aviao";

    groups[8] = vector<string>(2);
    groups[8][0] = "1";groups[8][1] = "2";

    groups[9] = vector<string>(6);
    groups[9][0] = "Verbo";groups[9][1] = "Pedra";groups[9][2] = "Junto";groups[9][3] = "Gasolina";groups[9][4] = "America";groups[9][5] = "Casa";

    groups[10] = vector<string>(2);
    groups[10][0] = "Lei";groups[10][1] = "Identidade";

    groups[11] = vector<string>(2);
    groups[11][0] = "D";groups[11][1] = "9";

    if(groupsidx.empty()){
        for(int i=0; i<groups.size(); i++){
            for(int j=0; j<groups[i].size(); j++){
                if(val == groups[i][j])
                    return i;
            }
        }
    }else{
        for(int i=0; i<groupsidx.size(); i++){
            for(int j=0; j<groups[groupsidx[i]].size(); j++){
                if(val == groups[groupsidx[i]][j])
                    return groupsidx[i];
            }
        }
    }
    return -1;
}

string num2Str(int num){
    switch (num){
        case 0:
            return "ZERO";
        case 1:
            return "UM";
        case 2:
            return "DOIS";
        case 3:
            return "TRES";
        case 4:
            return "QUATRO";
        case 5:
            return "CINCO";
        case 6:
            return "SEIS";
        case 7:
            return "SETE";
        case 8:
            return "OITO";
        case 9:
            return "NOVE";
        case 10:
            return "DEZ";
        case 11:
            return "ONZE";
        case 12:
            return "DOZE";
        default:
            return "menos um";

    }
}

void ImageProcessing(){
    string folderpath = "/home/gleidson/Documentos/NeuralNetwork/libra/Dataset/";
    vector<vector<float> > imgs_features;
    vector<vector<float> > shape_features;
    vector<string> groupsnames;
    vector<string> imgs_labels;
    vector<string> colnames;
    bool start = true;

    //para cada pasta
        //para cada imagem
            //obtem-se as características
                //se foi possível obter-las, salva as caracteríticas e o rótulo (nome da pasta com as imagens)
    vector<string> folders = getDirContent(folderpath);
    for(int l=0; l<folders.size(); l++){
        vector<string> folders2 = getDirContent(folderpath+folders[l]+"/");
        for(int i=0; i<folders2.size(); i++){
            vector<int> groupsidx;
            groupsidx.push_back(0);
            groupsidx.push_back(3);
            int groupOfSign = partOfGroup(folders2[i]);
            if(groupOfSign == -1)   continue;
            string filepath = folderpath+folders[l]+"/"+folders2[i]+"/";
            vector<string> files = getDirContent(filepath);
            for(int j=0; j<files.size(); j++){
                if (!strstr(files[j].c_str(),"c")){
                    string fullfilepath = filepath+files[j];
                    string fullfilepathseg = filepath+"c"+files[j];
                    Mat original = imread(fullfilepath.c_str(), CV_LOAD_IMAGE_COLOR);
                    Mat segmentada = imread(fullfilepathseg.c_str(), CV_LOAD_IMAGE_COLOR);
                    if( original.data != NULL && segmentada.data != NULL ){
                        //UTILIZAR HISTOGRAMA DE GRADIENTES
                        vector<float> histrow = hogDesc(original);
                        imgs_features.push_back(histrow);

                        //UTILIZAR DESCRITORES DE FORMA
                        vector<float> shaperow = shapeDesc(segmentada);
                        shape_features.push_back(shaperow);

                        if(start){
                            for(int k=0; k<histrow.size(); k++){
                                ostringstream tostr;
                                tostr << k;
                                colnames.push_back("hog"+tostr.str());
                            }
                            for(int k=0; k<shaperow.size(); k++){
                                ostringstream tostr;
                                tostr << k;
                                colnames.push_back("shp"+tostr.str());
                            }
                            colnames.push_back("groups");
                            colnames.push_back("label");
                            start = false;
                        }

                        //GRUPO
                        groupsnames.push_back(num2Str(groupOfSign));

                        //RÓTULO
                        imgs_labels.push_back(folders2[i]);

                    }
                }
            }
        }
    }
    //APENAS SHAPE E ROTULO
    /*for(int i = 0; i < shape_features.size(); i++){
    	for(int j = 0; j < shape_features[i].size(); j++){
    		cout << shape_features[i][j] << ";";
    	}
    	cout << imgs_labels[i] << endl;
    }*/

    //APENAS HISTOGRAMA E RÓTULO
    /*for(int i = 0; i < imgs_features.size(); i++){
    	for(int j = 0; j < imgs_features[i].size(); j++){
    		cout << imgs_features[i][j] << ";";
    	}
    	cout << imgs_labels[i] << endl;
    }*/

    //USAR DIAMBA AS PARTES :D
    for(int i = 0; i < colnames.size()-1; i++){
        cout << colnames[i] << ",";
    }
    cout << colnames[colnames.size()-1] << endl;

    for(int i = 0; i < imgs_labels.size(); i++){

        for(int j = 0; j < imgs_features[i].size(); j++){
    		cout << imgs_features[i][j] << ",";
    	}

        for(int j = 0; j < shape_features[i].size(); j++){
    		cout << shape_features[i][j] << ",";
    	}
        cout << groupsnames[i] << ",";
        cout << imgs_labels[i] << endl;


    }
}

int main()
{
    //Neural();
    ImageProcessing();
}
