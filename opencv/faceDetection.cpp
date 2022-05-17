#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;


////////////////// face detection with a model that use weights in xml already trained.//////////////////
int main()
{
    // woman face path
    string path = "dumpImageTest/test.png";

    // opencv read the img
    Mat inImg = imread(path);

    // now we call a model object that will use a xml weights
    CascadeClassifier faceCascade;

    // we load it
    faceCascade.load("weights/haarcascade_frontalface_default.xml");

    if (faceCascade.empty()) 
        cout << "XML file not loaded" <<endl;

    // and a vector of Rects, to take all the detection bounding boxes
    vector<Rect> faces;

    // here we detect
    faceCascade.detectMultiScale(inImg, faces, 1.1, 10);

    for(int i = 0; i < faces.size(); i++)
    {
        rectangle(inImg, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
    }

    imshow("outImg", inImg);
    waitKey(0);

    return 0;
}