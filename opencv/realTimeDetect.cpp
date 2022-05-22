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
    

    std::vector<std::string> class_names {"sign_red", "sign_green", "sign_yellow"};
    auto model = readNetFromDarknet("cfg.cfg", "weights/peso.weights");

    model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    model.setPreferableTarget(DNN_TARGET_OPENCL);


    while(true) 
    {
        Mat image;

        screen(image);

        cvtColor(image, image, COLOR_BGRA2BGR);

        int image_height = image.cols;
        int image_width = image.rows;

        auto start = getTickCount();
        
        Mat blob = blobFromImage(image, 0.00392, Size(320, 320), Scalar(0, 0, 0),true, false);
        
        model.setInput(blob);


        vector<String> names;
        vector<int> outLayers = model.getUnconnectedOutLayers();
        vector<String> layersNames = model.getLayerNames();

        names.resize(outLayers.size());

        for (size_t i =0; i < outLayers.size(); ++i)
        {
            names[i] = layersNames[outLayers[i] - 1];
        }

        vector<Mat> outMat;
        model.forward(outMat, names);

        cout << outMat[0] << endl;

        int a;
        cin >> a;

        float confThreshold = 0.20;
        vector<int> classIds;
        vector<float> confidences;
        vector<cv::Rect> boxes;
        for(size_t i = 0; i < outMat.size(); ++i)
        {
            float* data = (float*)outMat[i].data;
            for(int j=0; j < outMat[i].rows; ++j, data += outMat[i].cols)
            {
                cv::Mat scores = outMat[i].row(j).colRange(5, outMat[i].cols);
                cv::Point classId;
                double confidence;

                cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
                if (confidence > confThreshold)
                {
                    cv::Rect box; int cx, cy;
                    cx = (int)(data[0] * image.cols);
                    cy = (int)(data[1] * image.rows);
                    box.width = (int)(data[2] * image.cols);
                    box.height = (int)(data[3] * image.rows);
                    box.x = cx - box.width/2;
                    box.y = cy - box.height/2;

                    boxes.push_back(box);
                    classIds.push_back(classId.x);
                    confidences.push_back((float)confidence);
                }
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