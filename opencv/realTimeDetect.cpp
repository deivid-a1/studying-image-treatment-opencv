#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/ocl.hpp>

#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/dnn/all_layers.hpp>

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

vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

int main()
{


    cv::ocl::Context context;
    cv::ocl::Device(context.device(1));

    ScreenShot screen(0, 0, 1024, 768);

    Mat image, blob;
    UMat UImg, Ublob;
    

    std::vector<std::string> class_names {"sign_red", "sign_green", "sign_yellow"};
    auto model = readNetFromDarknet("cfg.cfg", "weights/peso.weights");

    model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    model.setPreferableTarget(DNN_TARGET_OPENCL);


    // cout << outNames[1] << endl;
    //     int a;
    //     cin >> a;

    while(true) 
    {
        // double start = clock();
        // printf("fps %.4f  spf %.4f\n", FPS(start), 1 / FPS(start));
        
        // current_ticks = clock();
        
        screen(image);

        cvtColor(image, image, COLOR_BGRA2BGR);

        int image_height = image.cols;
        int image_width = image.rows;

        auto start = getTickCount();
        
        blob = blobFromImage(image, 0.00392, Size(416, 416), Scalar(0, 0, 0),true, false);
        
        model.setInput(blob,"", 0.00392, 0);
    
        // vector<int> outLayers = model.getUnconnectedOutLayers();
        // vector<cv::String> layersNames = model.getLayerNames();
        
        // vector<cv::String> names;
        // names.resize(outLayers.size());

        // for(size_t i = 0; i < outLayers.size(); i++)
        // {
        //     names[i] = layersNames[outLayers[i] - 1];
        // }

        vector<Mat> outputs;
        vector<string> names = model.getUnconnectedOutLayersNames();

        Mat outMat;
        outMat = model.forward(names);

        // Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        for (int i = 0; i < outMat.rows; i++)
        {
            // int class_id = detectionMat.at<float>(i, 1);
            // float confidence = detectionMat.at<float>(i, 2);
            Mat scores = outMat.row(i).colRange(5, outMat.cols);
            Point PositionOfMax;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &PositionOfMax);
            
            
            // Check if the detection is of good quality
            if (confidence > 0.2){
                // int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
                // int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
                // int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
                // int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);
                // string class_name = class_names[class_id];
                // rectangle(image, Point(box_x, box_y), Point(box_x+box_width, box_y+box_height), Scalar(211,0,148), 2);
                // putText(image, class_name + " " + to_string(int(confidence*100)) + "%", Point(box_x, box_y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(211,0,148), 1);
                int centerX = (int)(outMat.at<float>(i, 0) * image.cols); 
                int centerY = (int)(outMat.at<float>(i, 1) * image.rows); 
                int width =   (int)(outMat.at<float>(i, 2) * image.cols+20); 
                int height =   (int)(outMat.at<float>(i, 3) * image.rows+100); 

                int left = centerX - width / 2;
                int top = centerY - height / 2;

                stringstream ss;
                ss << PositionOfMax.x;
                string clas = ss.str();
                int color = PositionOfMax.x * 10;
                putText(image, clas, Point(left, top), 1, 2, Scalar(color, 255, 255), 2, false);
                stringstream ss2;
                ss << confidence;
                string conf = ss.str();
                rectangle(image, Rect(left, top, width, height), Scalar(color, 0, 0), 2, 8, 0);
            }
        }   
        // UImg = img.getUMat(ACCESS_READ);

        imshow("image", image);
        
        char k = cv::waitKey(1);
        if (k == 'q')
            break;
        
        // delta_ticks = clock() - current_ticks; //the time, in ms, that took to render the scene
        // if(delta_ticks > 0)
        //     fps = CLOCKS_PER_SEC / delta_ticks;
        // cout << fps << endl;
    }

    return 0;
}