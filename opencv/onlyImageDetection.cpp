#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <time.h>
#include <fstream>
// #define FPS(start) (CLOCKS_PER_SEC / (clock()-start))

using namespace cv;
using namespace std;
using namespace dnn;


int main()
{

    vector<string> class_names;

    ifstream ifs(string("caffe/object_detection_classes_coco.txt").c_str());
    string line;

    while (getline(ifs, line))
    {
        cout << line << endl;
        class_names.push_back(line);
    } 

    auto model = readNet("caffe/frozen_inference_graph.pb","caffe/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt", "TensorFlow");
    // model.setPreferableTarget(DNN_TARGET_OPENCL);

    Mat image = imread("dumpImageTest/bunchTraffic.png");

    int image_height = image.cols;
    int image_width = image.rows;
    
    
    auto start = getTickCount();
    
    Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false);
    
    

    model.setInput(blob);


    
    Mat output = model.forward();



    auto end = getTickCount();

    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());


    
    for (int i = 0; i < detectionMat.rows; i++){
        int class_id = int(detectionMat.at<float>(i, 1));
        float confidence = detectionMat.at<float>(i, 2);

      
       // Check if the detection is of good quality
       if (confidence > 0.4){
           int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
           int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
           int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
           int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);
           rectangle(image, Point(box_x, box_y), Point(box_x+box_width, box_y+box_height), Scalar(211,0,148), 2);
           putText(image, class_names[class_id-1].c_str(), Point(box_x, box_y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
       }
   } 

    
    
    imshow("Image", image);
    waitKey(0);
    destroyAllWindows();

    


//     cv::ocl::Context context;
//     cv::ocl::Device(context.device(1));
    

//     std::vector<std::string> class_names = {"sign_red", "sign_green", "sign_yellow"};
//     auto model = readNet("weights/peso.weights","cfg.cfg");

//     model.setPreferableTarget(DNN_TARGET_OPENCL);

//     Mat image = imread("dumpImageTest/bunchTraffic.png");
//     int image_height = image.cols;
//     int image_width = image.rows;

//     Mat blob = blobFromImage(image, 0.00392, Size(320, 320), Scalar(0, 0, 0),true, false);

//     model.setInput(blob);

//     Mat output = model.forward();
//     Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

//     for (int i = 0; i < detectionMat.rows; i++){
//        int class_id = detectionMat.at<float>(i, 1);
//        float confidence = detectionMat.at<float>(i, 2);
      
//        // Check if the detection is of good quality
//        if (confidence > 0.4){
//            int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
//            int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
//            int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
//            int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);
//            rectangle(image, Point(box_x, box_y), Point(box_x+box_width, box_y+box_height), Scalar(255,255,255), 2);
//            putText(image, class_names[class_id-1].c_str(), Point(box_x, box_y-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
//        }
//    } 

//    imshow("image", image);
//    waitKey(0);
//    destroyAllWindows();

    return 0;
}


