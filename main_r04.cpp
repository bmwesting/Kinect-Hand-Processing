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

// conversion from cvConvexityDefect
struct ConvexityDefect
{
    Point start;
    Point end;
    Point depth_point;
    float depth;
};

// Thanks to Jose Manuel Cabrera for part of this C++ wrapper function
void findConvexityDefects(vector<Point>& contour, vector<int>& hull, vector<ConvexityDefect>& convexDefects)
{
    if(hull.size() > 0 && contour.size() > 0)
    {
        CvSeq* contourPoints;
        CvSeq* defects;
        CvMemStorage* storage;
        CvMemStorage* strDefects;
        CvMemStorage* contourStr;
        CvConvexityDefect *defectArray = 0;

        strDefects = cvCreateMemStorage();
        defects = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvSeq),sizeof(CvPoint), strDefects );

        //We transform our vector<Point> into a CvSeq* object of CvPoint.
        contourStr = cvCreateMemStorage();
        contourPoints = cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), contourStr);
        for(int i = 0; i < (int)contour.size(); i++) {
            CvPoint cp = {contour[i].x,  contour[i].y};
            cvSeqPush(contourPoints, &cp);
        }

        //Now, we do the same thing with the hull index
        int count = (int) hull.size();
        //int hullK[count];
        int* hullK = (int*) malloc(count*sizeof(int));
        for(int i = 0; i < count; i++) { hullK[i] = hull.at(i); }
        CvMat hullMat = cvMat(1, count, CV_32SC1, hullK);

        // calculate convexity defects
        storage = cvCreateMemStorage(0);
        defects = cvConvexityDefects(contourPoints, &hullMat, storage);
        defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*defects->total);
        cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);
        //printf("DefectArray %i %i\n",defectArray->end->x, defectArray->end->y);

        //We store defects points in the convexDefects parameter.
        for(int i = 0; i<defects->total; i++){
            ConvexityDefect def;
            def.start       = Point(defectArray[i].start->x, defectArray[i].start->y);
            def.end         = Point(defectArray[i].end->x, defectArray[i].end->y);
            def.depth_point = Point(defectArray[i].depth_point->x, defectArray[i].depth_point->y);
            def.depth       = defectArray[i].depth;
            convexDefects.push_back(def);
        }

    // release memory
    cvReleaseMemStorage(&contourStr);
    cvReleaseMemStorage(&strDefects);
    cvReleaseMemStorage(&storage);

    }
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
                    Scalar center = mean(contourMat);
                    Point centerPoint = Point(center.val[0], center.val[1]);

                    // approximate the contour by a simple curve
                    vector<Point> approxCurve;
                    approxPolyDP(contourMat, approxCurve, 10, true);

                    vector< vector<Point> > debugContourV;
                    debugContourV.push_back(approxCurve);
                    drawContours(rightHandDebug, debugContourV, 0, Scalar(0, 128, 0), 3);

                    vector<int> hull;
                    convexHull(Mat(approxCurve), hull, false, false);

                    // draw the hull points
                    for(int j = 0; j < hull.size(); j++)
                    {
                        int index = hull[j];
                        circle(rightHandDebug, approxCurve[index], 3, Scalar(0,128,200), 2);
                    }

                    //Convexity Defects Processing - TODO Later
                    vector<ConvexityDefect> convexDefects;
                    findConvexityDefects(approxCurve, hull, convexDefects);
                    printf("Number of defects: %d.\n", (int) convexDefects.size());

                    for(int j = 0; j < convexDefects.size(); j++)
                    {
                        circle(rightHandDebug, convexDefects[j].depth_point, 3, Scalar(128,155,200), 2);

                    }
                    
                    // assemble point set of convex hull
                    vector<Point> hullPoints;
                    for(int k = 0; k < hull.size(); k++)
                    {
                        int curveIndex = hull[k];
                        Point p = approxCurve[curveIndex];
                        hullPoints.push_back(p);
                    }

                    // area of hull and curve
                    double hullArea  = contourArea(Mat(hullPoints));
                    double curveArea = contourArea(Mat(approxCurve));
                    double handRatio = curveArea/hullArea;
                    //printf("Area of approxContour:     %f\n", curveArea);
                    //printf("Area of convexHull:        %f\n", hullArea);

                    // hand is grasping
                    if(handRatio > 0.8)
                        circle(rightHandDebug, centerPoint, 5, Scalar(0,255,0), 5);
                    else
                        circle(rightHandDebug, centerPoint, 5, Scalar(0,0,255), 5);
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
