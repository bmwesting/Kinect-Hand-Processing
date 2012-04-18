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

// median blur factor
const unsigned int MEDIAN_BLUR_K = 5;

// returns true if the hand is near the sensor area
bool handApproachingDisplayPerimeter(float x, float y)
{
    return (x > (XRES - ROI_OFFSET)) || (x < (ROI_OFFSET)) ||
           (y > (YRES - ROI_OFFSET)) || (y < (ROI_OFFSET));
}

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
                
                if(!handApproachingDisplayPerimeter(rightHand.x, rightHand.y))
                {
                    roi.x = rightHand.x - ROI_OFFSET;
                    roi.y = rightHand.y - ROI_OFFSET;
                    //printf("Hand depth = %d\n", handDepth);
                }
            }
        }
        else
            handDepth = -1;

        // update 16 bit depth matrix
        memcpy(depthRaw.data, sensor->getDepthData(), XRES*YRES*2);
        depthRaw.convertTo(depthShow, CV_8U, DEPTH_SCALE_FACTOR);

        // extract hand from image
        Mat rightHandCpy(depthShow, roi);
        Mat rightHand = rightHandCpy.clone();
         
        // binary threshold
        if(handDepth != -1)
            rightHand = (rightHand > (handDepth - BIN_THRESH_OFFSET)) & (rightHand < (handDepth + BIN_THRESH_OFFSET));

        // last pre-filtering step, apply median blur
        medianBlur(rightHand, rightHand, MEDIAN_BLUR_K);
        
        // create debug image of thresholded hand and cvt to RGB so hints show
        Mat rightHandDebug = rightHand.clone();
        cvtColor(rightHandDebug, rightHandDebug, CV_GRAY2RGB);

        std::vector< std::vector<Point> > contours;
        findContours(rightHand, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        if (contours.size()) {
            for (int i = 0; i < contours.size(); i++) {
                vector<Point> contour = contours[i];
                Mat contourMat = Mat(contour);
                double cArea = contourArea(contourMat);

                if(cArea > 2000) // likely the hand
                {
                    //printf("Area of contour[%d] = %f\n", i, area);
                    Scalar center = mean(contourMat);
                    Point centerPoint = Point(center.val[0], center.val[1]);
                    //circle(rightHandDebug, centerPoint, 5, Scalar(128,0,0), 5);
                    //drawContours(rightHandDebug, contours, i, Scalar(0, 128, 0), 3);
                    
                    // approximate the contour by a simple curve
                    vector<Point> approxCurve;
                    approxPolyDP(contourMat, approxCurve, 20, true);

                    vector<int> hull;
                    convexHull(Mat(approxCurve), hull);
                    
                    // draw the hull points
                    for(int j = 0; j < hull.size(); j++)
                    {
                        int index = hull[j];
                        circle(rightHandDebug, approxCurve[index], 3, Scalar(0,128,200), 2);
                    }
                    
                    //printf("Area of convex hull: %f\n", contourArea(Mat(hullContour)));

                    //approxPolyDP(Mat(hull), hullPoints, 0.001, true);
                    //printf("Area of convex hull: %f\n", contourArea(Mat(hullPoints)));
                    //if(hullArea/cArea > 0.8) 
                        //printf("grasping... hullArea/contourA = %f\n", hullArea/cArea);
                }
            }
        }

        resize(rightHandDebug, rightHandDebug, Size(), 3, 3);
        imshow("depthFrame", depthShow);
        imshow("handFrame",  rightHandDebug);

        key = waitKey(10);

    }

    delete sensor;

    return 0;
}
