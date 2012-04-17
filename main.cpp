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

// defines the value about which thresholding occurs
const unsigned int BIN_THRESH_OFFSET = 5;

// defines the value about witch the region of interest is extracted
const unsigned int ROI_OFFSET = 70;

int main(int argc, char** argv)
{

    // initialize the kinect
    sensor = new SkeletonSensor();
    sensor->initialize();
    sensor->setPointModeToProjective();

    Mat depthRaw(YRES, XRES, CV_16UC1);
    Mat depthShow(YRES, XRES, CV_8UC1);
    //Mat rightHand(ROI_OFFSET*2, ROI_OFFSET*2, CV_8UC1); // hand processing
    
    // rectangle used to extract hand regions from depth map
    Rect roi;
    roi.width  = ROI_OFFSET*2;
    roi.height = ROI_OFFSET*2;

    namedWindow("depthFrame", CV_WINDOW_AUTOSIZE);
    namedWindow("handFrame", CV_WINDOW_AUTOSIZE);

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
                roi.x = rightHand.x - ROI_OFFSET;
                roi.y = rightHand.y - ROI_OFFSET;
                printf("Hand depth = %d\n", handDepth);
            }
        }
        else
            handDepth = -1;

        // update 16 bit depth matrix
        memcpy(depthRaw.data, sensor->getDepthData(), XRES*YRES*2);
        depthRaw.convertTo(depthShow, CV_8U, DEPTH_SCALE_FACTOR);

        // extract hand from image
        Mat rightHand(depthShow, roi);
         
        // binary threshold
        if(handDepth != -1)
            rightHand = (rightHand > (handDepth - BIN_THRESH_OFFSET)) & (rightHand < (handDepth + BIN_THRESH_OFFSET));

       

        imshow("depthFrame", depthShow);
        imshow("handFrame",  rightHand);

        key = waitKey(10);

    }

    delete sensor;

    return 0;
}
