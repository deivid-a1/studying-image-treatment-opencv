#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <time.h>
// #define FPS(start) (CLOCKS_PER_SEC / (clock()-start))

using namespace cv;
using namespace std;
using namespace dnn;

class ScreenShot
{
    Display* display;
    Window root;
    int x,y,width,height;
    XImage* img{nullptr};
public:
    ScreenShot(int x, int y, int width, int height):
        x(x),
        y(y),
        width(width),
        height(height)
    {
        display = XOpenDisplay(nullptr);
        root = DefaultRootWindow(display);
    }

    void operator() (cv::Mat& cvImg)
    {
        if(img != nullptr)
            XDestroyImage(img);
        img = XGetImage(display, root, x, y, width, height, AllPlanes, ZPixmap);
        cvImg = cv::Mat(height, width, CV_8UC4, img->data);
    }

    ~ScreenShot()
    {
        if(img != nullptr)
            XDestroyImage(img);
        XCloseDisplay(display);
    }
};


double clockToMilliseconds(clock_t ticks){
    // units/(units/time) => time (seconds) * 1000 = milliseconds
    return (ticks/(double)CLOCKS_PER_SEC)*1000.0;
}