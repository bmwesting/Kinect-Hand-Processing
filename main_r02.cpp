#include "SkeletonSensor.h"

// openCV
#include <opencv/highgui.h>
#include <opencv/cv.h>
using namespace cv;

#include <iostream>
using namespace std;

// globals
SkeletonSensor* sensor;

const unsigned int XRES = 640;
const unsigned int YRES = 480;

const float DEPTH_SCALE_FACTOR = 255./4096.;

const unsigned int BIN_THRESH_OFFSET = 5;

int main(int argc, char** argv)
{

    // initialize the kinect
    sensor = new SkeletonSensor();
    sensor->initialize();

    Mat depthRaw(YRES, XRES, CV_16UC1);
    Mat depthShow(YRES, XRES, CV_8UC1);

    namedWindow("debugFrame", CV_WINDOW_AUTOSIZE);

    int key = 0;
    while(key != 27 && key != 'q')
    {

        sensor->waitForDeviceUpdateOnUser();
        
        int handDepth;
        if(sensor->getNumTrackedUsers() > 0)
        {
            Skeleton skel =  sensor->getSkeleton(sensor->getUID(0));
            SkeletonPoint rightHand = skel.rightHand;
            if(rightHand.confidence == 1.0)
            {
                handDepth = rightHand.z * (DEPTH_SCALE_FACTOR);
                printf("Hand depth = %d\n", handDepth);
            }
        }
        else
            handDepth = -1;

        // update 16 bit depth matrix
        memcpy(depthRaw.data, sensor->getDepthData(), XRES*YRES*2);
        depthRaw.convertTo(depthShow, CV_8U, DEPTH_SCALE_FACTOR);

        //static binary threshold
        if(handDepth != -1)
            depthShow = (depthShow > (handDepth - BIN_THRESH_OFFSET)) & (depthShow < (handDepth + BIN_THRESH_OFFSET));

        imshow("debugFrame", depthShow);

        key = waitKey(10);

    }

    delete sensor;

    return 0;
}
