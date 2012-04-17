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

		// update 16 bit depth matrix
        memcpy(depthRaw.data, sensor->getDepthData(), XRES*YRES*2);
	    depthRaw.convertTo(depthShow, CV_8U, 255/4096.0);

		printf("center depth value: %d\n",depthRaw.at<short>(XRES/2,YRES/2));

		imshow("debugFrame", depthShow);
		//imshow("colorFrame", rgb);

		key = waitKey(10);

	}

	delete sensor;

    return 0;
}
