#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


////////////// Draw Shapes and text ////////////
int main()
{
    /// here we are creating our own image, so the parameters represents X, Y, BGR format (standard format in opencv) and color based in format.
    Mat img(512, 512, CV_8UC3, Scalar(0, 0, 0));

    /// now we are creating a circle, its parameters is img input, center of the circle (X,Y), radius, color and line width (FILLED to color all the circle)
    circle(img, Point(int(512/2), int(512/2)), 155, Scalar(255, 0, 255), FILLED);

    ///it is the Rect() function, but we use Point() to represent the height and width;
    rectangle(img, Point(512/3,512/3), Point(512/2,512/2), Scalar(0, 0, 0), FILLED);

    /// Line draw
    line(img, Point(130, 296), Point(382, 310), Scalar(0,0,0),2);

    putText(img, "gg ez", Point(130, 292), FONT_HERSHEY_DUPLEX, 2, Scalar(0,0,0), 2);


    imshow("Image", img);
    waitKey(0);
    
    return 0;
}