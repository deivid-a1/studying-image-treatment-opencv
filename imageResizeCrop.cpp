#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


////////////// Resize and Crop function ////////////
int main()
{

    string path = "dumpImageTest/macaco.png";

    /// Mat é uma instância de um objeto dentro do OpenCV que cria matrizes
    Mat img = imread(path), imgResize, imgCrop;
    
    //cout << img.size() << endl;

    // resize the image with the scale (2 lasts arguments) or Size(X, Y) to set precisely.
    resize(img, imgResize, Size(), 0.5, 0.5);

    Rect roi(100, 100, 300, 300);
    imgCrop = img(roi);


    imshow("Image Resize", img);
    imshow("Image Crop", imgCrop);
    waitKey(0);
    
    return 0;
}