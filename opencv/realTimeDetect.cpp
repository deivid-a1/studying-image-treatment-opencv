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


vector<string> classes {"sign_green", "sign_red", "sign_yellow"};
float nms = 0.4;
float conf_threshold = 0.5;
vector<Mat> outs;

void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}


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

void remove_box(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > conf_threshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, conf_threshold, nms, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        draw_box(classIds[idx], confidences[idx], box.x, box.y,
        box.x + box.width, box.y + box.height, frame);
    }
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
    cv::ocl::Device(context);


cout << "Device Type: " << context.device(1).name()<< endl;
    int a;
    cin >> a;

    ScreenShot screen(0, 0, 1024, 768);
    
    auto model = readNetFromDarknet("cfg.cfg", "weights/peso.weights");

    model.setPreferableTarget(DNN_TARGET_OPENCL);


    while(true) 
    {
        Mat image;

        screen(image);

        cvtColor(image, image, COLOR_BGRA2BGR);

        auto start = getTickCount();
        
        Mat blob = blobFromImage(image, 0.00392, Size(416, 416), Scalar(0, 0, 0),true, false);
        
        model.setInput(blob);


        // vector<String> names;
        // vector<int> outLayers = model.getUnconnectedOutLayers();
        // vector<String> layersNames = model.getLayerNames();

        // names.resize(outLayers.size());

        // for (size_t i =0; i < outLayers.size(); ++i)
        // {
        //     names[i] = layersNames[outLayers[i] - 1];
        // }

        vector<Mat> outMat;
        model.forward(outMat, getOutputsNames(model));

        auto end = getTickCount();

        remove_box(image, outMat);

        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = model.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(image, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2, false);

    
        // UImg = img.getUMat(ACCESS_READ);

        auto totalTime = (end - start) / getTickFrequency();

        putText(image, "FPS: " + to_string(int(1/totalTime)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0,255,0), 2, false);

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


