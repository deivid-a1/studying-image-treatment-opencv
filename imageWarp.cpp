#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


///the main objective is to distort the monkey image, the result is to seems like the monkey is seeing to the right, but in the real image it is seeing to the front.
int main()
{
    string path = "dumpImageTest/macaco.png";

    Mat img = imread(path);

    /// resizing the image, to a known value;
    resize(img, img, Size(700,700));

    /// now that we know the value, we can choose four point in the image to distorce. I choose the face of the monkey.
    Point2f src[4] = {{700/3, 700/3},{700 -(700/3), 700/3},{700/3, 700-(700/3)}, {700-(700/3),700-(700/3)}};
    /// now we use math to distorce the image in a diferent size, here is the out point.
    Point2f dst[4] = {{20.0f, 20.0f}, {250.0f,0.0f}, {0.0f,350.f}, {230.f, 330.f}};

    Mat matrix, imgWarp;

    /// use this function that will distorce the image together with the next one;
    matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, imgWarp, matrix, Point(247, 347));
    
    imshow("Image", img);
    imshow("Image Warp", imgWarp);

    waitKey(0);

    return 0;
}