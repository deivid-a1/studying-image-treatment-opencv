#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

///////////////// some function//////////////////
int main()
{

    string path = "dumpImageTest/macaco.png";

    /// Mat é uma instância de um objeto dentro do OpenCV que cria matrizes
    Mat img = imread(path);

    Mat imgGray, imgBlur, imgCanny, imgDia, imgErode;

    ///img is the input image, imgGray is the output, the third argument represents a COLOR (this is a INTEGER like 1, 2, 3...), 
    ///opencv have already a bunch of CONSTANTS that represent a log of colors like COLOR_BGR2GRAY.
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    
    ///used to reduce detais img, tranfering to imgBlur Mat object;
    GaussianBlur(img, imgBlur, Size(3,3), 5, 0);

    ///Filter, in general used with a GaussianBlur filter;
    Canny(imgBlur, imgCanny, 25, 75);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));

    dilate(imgCanny, imgDia, kernel);

    erode(imgDia, imgErode, kernel);

    ///imshow("Image", img);
    ///imshow("Image Gray", imgGray);
    ///imshow("Image Blur", imgBlur);
    imshow("Image Canny", imgCanny);
    imshow("Image Dilate", imgDia);
    imshow("Image Erode", imgErode);
    waitKey(0);
    
    return 0;
}

///g++ imageProcessing.cpp -o compilation/imagePGray -std=c++11 `pkg-config --cflags --libs opencv` to compile