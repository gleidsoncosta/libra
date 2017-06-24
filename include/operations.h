#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <math.h>


#include <string>
#include <iostream>
#include <math.h>
#include <vector>

#define PI 3.14
/***************************************
Processamento de Imagens
Gleidson Mendes Costa       31/05/2016
SO: Ubuntu 12, processador: intel i3
Operacao em imagens
****************************************/

/***************************************
Biblioteca com declaração de todas as classes
do programa
***************************************/

using namespace std;  //sistema
using namespace cv; //opencv

uchar notOp(uchar value, int bits_max);
uchar orOp(uchar value1, uchar value2);
uchar andOp(uchar value1, uchar value2);

class Operacoes{

    public:
        static Mat readImg(string filepath);
        static Mat invertImage(Mat img);

        static Mat quantizacaoImg(Mat img, int from_bits, int to_bits, bool grad);
        static Mat quantizacaoImgColor(Mat img, int from_bits, int to_bits);
        static Mat binImageThereshold(Mat img, int threshold);
        static Mat binImageYUV(Mat img);
        static Mat binImageHSV(Mat img);
        static Mat diferencaImg(Mat img1, Mat img2, int thres);
        static Mat diferencaImgColored(Mat img1, Mat img2, int thres);
        static Mat showHandSegmentada(Mat img, Mat thres);

        static void movementDiff(Mat img1, Mat img2);

        static Mat translateImg(Mat img, int x, int y);
        static Mat rotateImg(Mat img, int c_x, int c_y, double angle, double scale);
        static Mat resizeImg(Mat img, double x, double y);

        static Mat logicNOT(Mat img, int bits);
        static Mat logicOR(Mat img1, Mat img2, int bits);

        static Mat logicAND(Mat img1, Mat img2, int bits);
        static Mat logicAND(Mat img1, int value, int bits);
        static Mat logicFilter(Mat back, Mat skin);

        static Mat logicXOR(Mat img1, Mat img2, int bits);
        static Mat logicSUB(Mat img1, Mat img2, int bits);

        static Mat Times(Mat img1, float alpha, int bits);

};

class Histograma{

    public:
        int bits;
        int num_colors;
        vector<int> gray_pallete;
        vector<int> histograma;
        vector<int> histograma_acumulado;
        vector<int> new_histograma;
        Mat img;

        void initializeHistograma(Mat image, int b);
        void makeHistograma();
        void makeHistogramaAcumulado();
        void makeHistogramaNormalizado();
        Mat equalizeImg();
        Mat stretchingImg();

        void printHistograma(vector<int> hist);

};

class Filtros{

    public:

        static Mat midFilter(Mat img, int size);
        static Mat passAltaFilt(Mat img, int size, int val);
        static Mat unsharpMask(Mat img1, Mat img2);
        static Mat gaussFilter(Mat img, float sigma, int size);
        static Mat ditheringImage(Mat img);
        static Mat ditheringAperiodic(Mat img);
        static Mat maxPooling(Mat img);
};

class BackSub{
    public:
        Mat mask;
        Ptr<BackgroundSubtractor> pMOG;

        int actual_slot;
        int size_mov;
        int diff;
        bool first;
        Mat aux;
        Mat resp;

        Mat mov_variation[3];
        Mat mov;

        BackSub();
        void updateMog(Mat img);
        void updateMov(Mat img);
        Mat calcVariation(Mat act, Mat old);

};

class Calcs{
    public:
        static vector<float> vecMult(vector<float>v1, vector<float>v2);
        static float dotProduct(vector<float>v1, vector<float>v2);
        static float vecNorma(vector<float>v1);

        static float calcNumerador(vector<float>v1, vector<float>v2);
        static float calcDenominador(vector<float>v1, vector<float>v2);

        static float calcTotal(vector<float>p1, vector<float>p, vector<float>p2);

        static Point centroid(vector<Point> p );
        static float innerAngle(Point p1, Point p2, Point c);
        static float eucDistance(Point p1, Point p2);
};
