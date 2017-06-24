
#include "../include/operations.h"

/***************************************
Processamento de Imagens
Gleidson Mendes Costa       24/05/2016
SO: Ubuntu 12, processador: intel i3
Operacao em imagens
****************************************/

/***************************************
Funcionalidades do sistema desenvolvido
***************************************/


/***************************************
Leitura de uma imagem
Entrada: o caminho do arquivo
Saida: Imagem em um objeto Mat ou NULL se nao existir
***************************************/
Mat Operacoes::readImg(string file_path){
    Mat img;
    img = imread(file_path);   // Read the file

    // Check for invalid input
    if(! img.data ){
        img = NULL;
    }
    return img;
}

/***************************************
Inverte a imagem no eixo y
Entrada: Imagem
Saida: Imagem invertida
***************************************/
Mat Operacoes::invertImage(Mat img){
    Mat result(img.rows, img.cols, CV_8UC1);
    for(int i = 0; i < result.rows; i++){
        for(int j=0; j< result.cols; j++){
            uchar val = img.at<uchar>(i, j);
            result.at<uchar>(i, (result.cols-1)-j) = val;
        }
    }

    return result;
}

/***************************************
Quantiza a imagem
Entrada: Imagem, Numero de bits utilizados, Numero de bits desejados, identificador
se quer uma reducao gradual
Saida: Imagem quantizada
***************************************/
Mat Operacoes::quantizacaoImg(Mat img, int from_bits, int to_bits, bool grad){
    Mat result;
    img.copyTo(result);
    int steps = 0 ;
    int dim = 1;

    if(!grad){
        steps = 1;
    }else{
        steps = (int)(from_bits - to_bits)/dim;
    }

    for(int i=0; i<steps; i++){
        int num_bits;
        int new_num_bits;
        if(grad){
            num_bits = from_bits - (i*dim);
            new_num_bits = from_bits - ((i+1)*dim);
        }else{
            num_bits = from_bits;
            new_num_bits = to_bits;
        }
        for(int i = 0; i < result.rows; i++){
            for(int j=0; j< result.cols; j++){
                uchar val = result.at<uchar>(i, j);
                result.at<uchar>(i, j) = (int)((val*pow(2, new_num_bits))/pow(2, num_bits));
            }
        }
    }

    return result;
}

Mat Operacoes::quantizacaoImgColor(Mat img, int from_bits, int to_bits){
    Mat result;
    img.copyTo(result);
    int steps = 1 ;
    int dim = 1;

    for(int i=0; i<steps; i++){
        int num_bits;
        int new_num_bits;
            num_bits = from_bits - (i*dim);
            new_num_bits = from_bits - ((i+1)*dim);
        for(int j = 0; j < result.rows; j++){
            for(int k=0; k< result.cols; k++){
                for(int l=0; l<3; l++){
                    int val = (int)result.at<cv::Vec3b>(j, k)[l];
                    result.at<cv::Vec3b>(j, k)[l] = (int)((val*pow(2, new_num_bits))/pow(2, num_bits));
                }
            }
        }
    }

    return result;
}

Mat Operacoes::binImageThereshold(Mat img, int threshold){
    Mat result(img.rows, img.cols, CV_8UC1);

    for(int i=0; i<img.rows; i++){
        for(int  j=0; j<img.cols; j++){
            if(img.at<uchar>(i, j) < threshold){
                result.at <uchar>(i, j) = 0;
            }else{
                result.at <uchar>(i, j) = 255;
            }
        }
    }

    return result;
}

Mat Operacoes::binImageYUV(Mat img){
    Mat result(img.rows, img.cols, CV_8UC1);

    for(int i=0; i<img.rows; i++){
        for(int  j=0; j<img.cols; j++){
            if((img.at<cv::Vec3b>(i, j)[1] >= 77 && img.at<cv::Vec3b>(i, j)[1] <= 127) &&
                 (img.at<cv::Vec3b>(i, j)[2] >= 133 && img.at<cv::Vec3b>(i, j)[2] <= 173)){
                result.at <uchar>(i, j) = 0;
            }else{
                result.at <uchar>(i, j) = 255;
            }
        }
    }

    return result;
}

Mat Operacoes::binImageHSV(Mat img){
    Mat result(img.rows, img.cols, CV_8UC1);

    for(int i=0; i<img.rows; i++){
        for(int  j=0; j<img.cols; j++){
            if((img.at<cv::Vec3b>(i, j)[2] >= 40) &&
               (img.at<cv::Vec3b>(i, j)[0] <= (((-0.4*(img.at<cv::Vec3b>(i, j)[2]))+75))) &&
                 (img.at<cv::Vec3b>(i, j)[2] >= 133 && img.at<cv::Vec3b>(i, j)[2] <= 173)){
                result.at <uchar>(i, j) = 0;
            }else{
                result.at <uchar>(i, j) = 255;
            }
        }
    }

    return result;
}

Mat Operacoes::showHandSegmentada(Mat img, Mat thres){
    Mat resp;
    img.copyTo(resp);

    int c =0;
    int b = 0;
    for(int i=0; i<thres.rows; i++){
        for(int j=0; j<thres.cols; j++){
            int val = (int)thres.at<uchar>(i, j);

            if(val == 0){
                c++;
                resp.at<cv::Vec3b>(i, j)[0] = 0;
                resp.at<cv::Vec3b>(i, j)[1] = 0;
                resp.at<cv::Vec3b>(i, j)[2] = 0;
            }else{
b++;
                resp.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0];
                resp.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1];
                resp.at<cv::Vec3b>(i, j)[2] = img.at<cv::Vec3b>(i, j)[2];
            }
        }
    }
    cout << c << endl;
    cout << b << endl;
    return resp;
}

/***************************************
Subtrai uma imagem de outra
Entrada: Imagem 1, Imagem 2, e a Imagem para salvar o resultado
***************************************/
Mat Operacoes::diferencaImg(Mat a1, Mat a2, int thres){
    Mat img1, img2;
    cvtColor(a1, img1, COLOR_BGR2GRAY);
    cvtColor(a2, img2, COLOR_BGR2GRAY);

    Mat diff(img1.rows, img1.cols, CV_8UC1);

    for(int i = 0; i < diff.rows; i++){
        for(int j=0; j< diff.cols; j++){
            int val = abs((int)(img2.at<uchar>(i, j) - img1.at<uchar>(i, j)));
            if(val >= thres){
                diff.at<uchar>(i, j) = 255;
            }else{
                diff.at<uchar>(i, j) = 0;
            }
        }
    }

    /*int max = diff.at<uchar>(0, 0);
    int min = diff.at<uchar>(0, 0);

    for(int i = 0; i < diff.rows; i++){
        for(int j=0; j< diff.cols; j++){
            if(max < diff.at<uchar>(i, j)){
                max = diff.at<uchar>(i, j);
            }
            if(min > diff.at<uchar>(i, j)){
                min = diff.at<uchar>(i, j);
            }
        }
    }

    cout << "maximo "<< max << endl;
    cout << "minimo "<< min << endl;
*/
    return diff;
}

Mat Operacoes::diferencaImgColored(Mat img1, Mat img2, int thres){
    Mat diff(img1.rows, img1.cols, CV_8UC1);
    for(int i = 0; i < diff.rows; i++){
        for(int j=0; j< diff.cols; j++){
            bool r = false;
            bool g = false;
            bool b = false;
            for(int k=0; k<3; k++){
                uchar val = abs((img1.at<cv::Vec3b>(i, j)[k] - img2.at<cv::Vec3b>(i, j)[k]));

                /*if ( val > thres){
                    if(k == 0){
                        r = true;
                    }else if(k == 1){
                        g = true;
                    }else if(k == 2){
                        b = true;
                    }
                }*/
                diff.at<uchar>(i, j) = val;
            }
            /*if ( r && g && b){
                diff.at<uchar>(i, j) = 255;
            }else{
                diff.at<uchar>(i, j) = 0;
            }*/
        }
    }
    return diff;
}

void Operacoes::movementDiff(Mat img1, Mat img2){
    int mat[img1.rows][img1.cols];
    for(int i=0; i<img1.rows; i++){
        for(int j=0; j<img1.cols; j++){
            if(img1.at<uchar>(i,j) != img2.at<uchar>(i,j)){
                //if(img1.at<uchar>(i,j) == 0)
            }
            mat[i][j] = abs((int)(img1.at<uchar>(i, j) - img2.at<uchar>(i, j)));
        }
    }
    vector<int> num;

    for(int i=0; i<img1.rows; i++){
        for(int j=0; j<img1.cols; j++){
            int val = mat[i][j];
            if(num.size() >= 0){
                bool no= false;
                for(int k=0; k<num.size(); k++){
                    if(val == num[k]){
                        no = true;
                        break;
                    }
                }
                if(!no){
                    num.push_back(val);
                }
            }
        }
    }

    cout << "----------------" << endl;
    for(int i=0; i<num.size(); i++){
        cout << num[i] << " | ";
    }
}
/***************************************
Move a imagem
Entrada: Imagem, distancia em x para mover, distancia em y para mover
Saida: Imagem movida
***************************************/
Mat Operacoes::translateImg(Mat img, int x, int y){
    Mat result  = (Mat_ <double>(2,3) << 1, 0, x, 0, 1, y);
    Mat dst;
    warpAffine(img, dst, result, img.size());

    return dst;
}

/***************************************
Rotaciona a Imagem
Entrada: Imagem, centro em x da imagem, centro y da imagem, angulo de rotacao, escala
Saida: Imagem rotacionada
***************************************/
Mat Operacoes::rotateImg(Mat img, int c_x, int c_y, double angle, double scale){
    Mat result;
    Mat dst;
    Point2f center(c_x, c_y);

    result = getRotationMatrix2D(center, angle, scale);
    warpAffine(img, dst, result, Size(img.cols, img.rows));

    return dst;
}

/***************************************
Escalona a imagem
Entrada: Imagem, fator de escalonamento em x, fator de escalonamento em y
Saida: Imagem escalada
***************************************/
Mat Operacoes::resizeImg(Mat img, double x, double y){
    Mat result;
    Mat dst;
    resize(img, dst, Size(0, 0), x, y);

    return dst;
}

/***************************************
Operacao Logica Not
Entrada: Imagem, num de bits da imagem
Saida: Imagem trabalhada
***************************************/
Mat Operacoes::logicNOT(Mat img, int bits){
    Mat result;
    img.copyTo(result);
    int bit_val = pow(2, bits)-1;

    for(int i = 0; i < result.rows; i++){
        for(int j=0; j< result.cols; j++){
            uchar val = result.at<uchar>(i, j);
            result.at<uchar>(i, j) = notOp(val, bit_val);
        }
    }

    return result;
}

/***************************************
Operacao logica OR
Entrada: Imagem 1, Imagem 2, num de bits das imagens
Saida: Imagem trabalhada
***************************************/
Mat Operacoes::logicOR(Mat img1, Mat img2, int bits){
    Mat result;
    img1.copyTo(result);
    int bit_val = pow(2, bits)-1;

    for(int i = 0; i < result.rows; i++){
        for(int j=0; j< result.cols; j++){
            uchar val = orOp(result.at<uchar>(i, j), img2.at<uchar>(i,j));
            if(val > bit_val)   val = bit_val;
            result.at<uchar>(i, j) = val;
        }
    }

    return result;
}

/***************************************
Operacao logica AND
Entrada: Imagem 1, Imagem 2, num de bits das imagens
Saida: Imagem trabalhada
***************************************/
Mat Operacoes::logicAND(Mat img1, Mat img2, int bits){
    Mat result;
    img1.copyTo(result);
    int bit_val = pow(2, bits)-1;

    for(int i = 0; i < result.rows; i++){
        for(int j=0; j< result.cols; j++){
            uchar val = andOp(result.at<uchar>(i, j), img2.at<uchar>(i,j));
            if(val > bit_val)   val = bit_val;
            result.at<uchar>(i, j) = val;
        }
    }

    return result;
}

Mat Operacoes::Times(Mat img1, float alpha, int bits){
    Mat result;
    img1.copyTo(result);
    int bit_val = pow(2, bits)-1;

    for(int i = 0; i < result.rows; i++){
        for(int j=0; j< result.cols; j++){
            uchar val = result.at<uchar>(i, j) * alpha;
            if(val > bit_val)   val = bit_val;
            result.at<uchar>(i, j) = val;
        }
    }

    return result;
}

/***************************************
Operacao logica AND para escalares
Entrada: Imagem 1, valor escalar, num de bits das imagens
Saida: Imagem trabalhada
***************************************/
Mat Operacoes::logicAND(Mat img1, int value, int bits){
    Mat result;
    img1.copyTo(result);
    int bit_val = pow(2, bits)-1;

    for(int i = 0; i < result.rows; i++){
        for(int j=0; j< result.cols; j++){
            uchar val = andOp(result.at<uchar>(i, j), value);
            if(val > bit_val)   val = bit_val;
            result.at<uchar>(i, j) = val;
        }
    }

    return result;
}

Mat Operacoes::logicFilter(Mat back, Mat skin){
    Mat result(back.rows, back.cols, CV_8UC1);;

    for(int i = 0; i < result.rows; i++){
        for(int j=0; j< result.cols; j++){
            if(back.at<uchar>(i,j) == 0){
                result.at<uchar>(i,j) = 0;
            }else{
                if(skin.at<uchar>(i,j) == 0){
                    result.at<uchar>(i,j) = 0;
                }else{
                    result.at<uchar>(i,j) = 255;
                }
            }
        }
    }

    return result;
}

/***************************************
Operacao logica XOR
Entrada: Imagem 1, Imagem 2, num de bits das imagens
Saida: Imagem trabalhada
***************************************/
Mat Operacoes::logicXOR(Mat img1, Mat img2, int bits){
    Mat result;
    img1.copyTo(result);
    int bit_val = pow(2, bits)-1;

    for(int i = 0; i < result.rows; i++){
        for(int j=0; j< result.cols; j++){
            //or(and(a, not(b)), and(not(a), b))
            uchar val = orOp(andOp(img1.at<uchar>(i, j) , notOp(img2.at<uchar>(i, j), bit_val)),
            andOp(notOp(img1.at<uchar>(i, j), bit_val) , img2.at<uchar>(i, j)));
            if(val > bit_val)   val = bit_val;
            result.at<uchar>(i, j) = val;
        }
    }

    return result;
}

/***************************************
Operacao logica SUB
Entrada: Imagem 1, Imagem 2, num de bits das imagens
Saida: Imagem trabalhada
***************************************/
Mat Operacoes::logicSUB(Mat img1, Mat img2, int bits){
    Mat result;
    img1.copyTo(result);
    int bit_val = pow(2, bits)-1;

    for(int i = 0; i < result.rows; i++){
        for(int j=0; j< result.cols; j++){
            //and(a, not(b))
            uchar val = andOp(img1.at<uchar>(i, j) , notOp(img2.at<uchar>(i, j), bit_val));
            if(val > bit_val)   val = bit_val;
            result.at<uchar>(i, j) = val;
        }
    }

    return result;
}

/***************************************
Troca da cor do pixel pelo escalar
Entrada: valor 1, valor do maior numero de bit
Saida: resultado
***************************************/
uchar notOp(uchar value, int bits_max){
    return bits_max - value;
}

/***************************************
Soma Ṕixel
Entrada: valor 1, valor 2
Saida: resultado
***************************************/
uchar orOp(uchar value1, uchar value2){
    return value1 + value2;
}

/***************************************
Multiplicacao de pixel
Entrada: valor 1, valor 2
Saida: resultado
***************************************/
uchar andOp(uchar value1, uchar value2){
    return value1 * value2;
}
