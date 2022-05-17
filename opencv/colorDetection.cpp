#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/////// Color detection \\\\\\\/
int main()
{
    string path = "dumpImageTest/skyline.png";
    Mat inImg, outImg, mask;

    inImg = imread(path);

    /// with this configuration we can detect only the car in the image that have blue colors.
    int hmin= 91, smin = 179, vmin = 56,hmax = 255,smax = 34,vmax = 255;


    /// the configuration was found through this trackbars
    namedWindow("Trackbars", (640,200));
    createTrackbar("Hue Min", "Trackbars", &hmin, 179);
    createTrackbar("Hue Max", "Trackbars", &hmax, 179);
    createTrackbar("Sat Min", "Trackbars", &smin, 255);
    createTrackbar("Sat Max", "Trackbars", &smax, 255);
    createTrackbar("Val Min", "Trackbars", &vmin, 255);
    createTrackbar("Val Max", "Trackbars", &vmax, 255);

    while(true)
    {
        Scalar lower(hmin, smin, vmin);
        Scalar upper(hmax, smax, vmax);

        cvtColor(inImg, outImg, COLOR_BGR2HSV);

        inRange(outImg, lower, upper, mask);


        imshow("inImg", inImg);
        imshow("imgOut", outImg);
        imshow("imgMask", mask);
        waitKey(1);
    }
    

    return 0;
}