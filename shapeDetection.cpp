#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat getContours(Mat inImg, Mat outImg)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat exitImg = outImg;
    findContours(inImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    drawContours(exitImg, contours, -1, Scalar(255,0,255), 2);

    return exitImg;
}

int main()
{
    string path = "dumpImageTest/shapes.png";

    Mat inImg = imread(path);

    Mat imgGray, imgBlur, imgCanny, imgDia, imgErode;

    ///img is the input image, imgGray is the output, the third argument represents a COLOR (this is a INTEGER like 1, 2, 3...), 
    ///opencv have already a bunch of CONSTANTS that represent a log of colors like COLOR_BGR2GRAY.
    cvtColor(inImg, imgGray, COLOR_BGR2GRAY);
    
    ///used to reduce detais img, tranfering to imgBlur Mat object;
    GaussianBlur(inImg, imgBlur, Size(3,3), 3, 0);

    ///Filter, in general used with a GaussianBlur filter;
    Canny(imgBlur, imgCanny, 25, 75);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));

    dilate(imgCanny, imgDia, kernel);

    Mat exitImg = getContours(imgDia, inImg);

    //imshow("Image", inImg);
    /*imshow("Image Gray", imgGray);
    imshow("Image Blur", imgBlur);
    imshow("Image Canny", imgCanny);
    */
    imshow("Image Dilate", imgDia);
    imshow("Image In", inImg);
    waitKey(0);


    return 0;
}