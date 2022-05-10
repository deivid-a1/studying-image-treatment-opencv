#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
/*//////////////////// image //////////////////////
int main()
{

    string path = "dumpImageTest/macaco.png";

    /// Mat é uma instância de um objeto dentro do OpenCV que cria matrizes
    Mat img = imread(path);
    
    imshow("Image", img);
    waitKey(0);
    
    return 0;
}
*/////////////////////  Video ///////////////////

int main() 
{
    string path = "dumpImageTest/traffic.mp4";
    VideoCapture cap(path);

    Mat img;

    while (true)
    {
        cap.read(img);

        imshow("Image", img);
        waitKey(20);
    }

    return 0;
}