
#ifndef SKELETON_SENSOR_H
#define SKELETON_SENSOR_H

#include <XnCppWrapper.h>
#include <vector>

// A 3D point with the confidence of the point's location. confidence_ > 0.5 is good
struct SkeletonPoint
{
    float x, y, z, confidence;
};

struct Skeleton
{
    SkeletonPoint head;
    SkeletonPoint neck;
    SkeletonPoint rightShoulder;
    SkeletonPoint leftShoulder;
    SkeletonPoint rightElbow;
    SkeletonPoint leftElbow;
    SkeletonPoint rightHand;
    SkeletonPoint leftHand;
    SkeletonPoint rightHip;
    SkeletonPoint leftHip;
    SkeletonPoint rightKnee;
    SkeletonPoint leftKnee;
    SkeletonPoint rightFoot;
    SkeletonPoint leftFoot;
    SkeletonPoint torso;

};

// SkeletonSensor: A wrapper for OpenNI Skeleton tracking devices
//
// Requires the OpenNI + NITE framework installation and the device driver
// Tracks users within the device FOV, and assists in collection of user joints data
class SkeletonSensor
{
    public:
        SkeletonSensor();
        ~SkeletonSensor();

        // set up the device resolution and data generators
        int initialize();

        // non-blocking wait for new data on the device
        void waitForDeviceUpdateOnUser();

        // update vector of tracked users
        void updateTrackedUsers();

        // return true if UID is among the tracked users
        bool isTracking(const unsigned int uid);

        // returns skeleton of specified user
        Skeleton getSkeleton(const unsigned int uid);

        // returns vector of skeletons for all users
        std::vector<Skeleton> getSkeletons();

        // get number of tracked users
        unsigned int getNumTrackedUsers();

        // map tracked user index to UID
        unsigned int getUID(const unsigned int index);

        // change point mode
        void setPointModeToProjective();
        void setPointModeToReal();
        
        // get depth and image data
        const XnDepthPixel* getDepthData();
        const XnDepthPixel* getWritableDepthData(){};
        const XnUInt8* getImageData();
        const XnLabel*     getLabels();

    private:
        xn::Context context_;
        xn::DepthGenerator depthG_;
        xn::UserGenerator userG_;
        xn::ImageGenerator imageG_;

        std::vector<unsigned int> trackedUsers_;
        
        // current list of hands
        //std::list<XnPoint3D> handCursors;

        bool pointModeProjective_;

        // on user detection and calibration, call specified functions
        int setCalibrationPoseCallbacks();

        // joint to point conversion, considers point mode
        void convertXnJointsToPoints(XnSkeletonJointPosition* const j, SkeletonPoint* const p, unsigned int numPoints);

        // callback functions for user and skeleton calibration events
        static void XN_CALLBACK_TYPE newUserCallback(xn::UserGenerator& generator, XnUserID nId, void* pCookie);
        static void XN_CALLBACK_TYPE lostUserCallback(xn::UserGenerator& generator, XnUserID nId, void* pCookie);
        static void XN_CALLBACK_TYPE calibrationStartCallback(xn::SkeletonCapability& capability, XnUserID nId, void* pCookie);
        static void XN_CALLBACK_TYPE calibrationCompleteCallback(xn::SkeletonCapability& capability, XnUserID nId, XnCalibrationStatus eStatus, void* pCookie);
        static void XN_CALLBACK_TYPE poseDetectedCallback(xn::PoseDetectionCapability& capability, const XnChar* strPose, XnUserID nId, void* pCookie);
};

#endif
