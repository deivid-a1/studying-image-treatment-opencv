#include <opencv2/core/ocl.hpp>
#include <iostream>


using namespace std;

int check()
{
    cv::ocl::Device device = cv::ocl::Device::getDefault();
    
    cout << device.extensions() << endl;

    return 1;
}

int main()
{
    check();

    return 0;
}
