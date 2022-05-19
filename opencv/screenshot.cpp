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



int main()
{


    cv::ocl::Context context;
    cv::ocl::Device(context.device(1));

    ScreenShot screen(0, 0, 1024, 768);

    Mat image, blob;
    UMat UImg, Ublob;
    

    std::vector<std::string> class_names = {"sign_red", "sign_green", "sign_yellow"};
    auto model = readNet("weights/peso.weights","cfg.cfg");

    model.setPreferableTarget(DNN_TARGET_OPENCL);

    clock_t current_ticks, delta_ticks;
    clock_t fps = 0;
    

    while(true) 
    {
        // double start = clock();
        // printf("fps %.4f  spf %.4f\n", FPS(start), 1 / FPS(start));
        
        current_ticks = clock();
        
        // screen(image);
        image = imread("dumpImageTest/traffic.png");

        // cvtColor(image, image, COLOR_BGRA2BGR);

        
        blob = blobFromImage(image, 0.00392, Size(320, 320), Scalar(0, 0, 0),true, false);

    

        int image_height = image.cols;
        int image_width = image.rows;
        
        model.setInput(blob);
        
    
        Mat output = model.forward();

        
        Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());


        

        for (int i = 0; i < detectionMat.rows; i++)
        {
            int class_id = detectionMat.at<float>(i, 1);
            float confidence = detectionMat.at<float>(i, 2);
      
            // Check if the detection is of good quality
            if (confidence > 0.2){
                int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
                int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
                int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
                int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);
                rectangle(image, Point(box_x, box_y), Point(box_x+box_width, box_y+box_height), Scalar(255,255,255), 2);
                putText(image, class_names[class_id-1].c_str(), Point(box_x, box_y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
            }
         
        }   
        // UImg = img.getUMat(ACCESS_READ);

        imshow("image", image);
        

        
        char k = cv::waitKey(1);
        if (k == 'q')
            break;
        
        delta_ticks = clock() - current_ticks; //the time, in ms, that took to render the scene
        if(delta_ticks > 0)
            fps = CLOCKS_PER_SEC / delta_ticks;
        cout << fps << endl;
    }

    return 0;
}