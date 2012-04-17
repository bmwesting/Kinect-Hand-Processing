
#include "SkeletonSensor.h"
#include "log.h"

inline int CHECK_RC(const unsigned int rc, const char* const description)
{
    if(rc != XN_STATUS_OK)
    {
        put_flog(LOG_ERROR, "%s failed: %s", description, xnGetStatusString(rc));
        return -1;
    }

    return 0;
}

SkeletonSensor::SkeletonSensor()
{
    pointModeProjective_ = false;
}

SkeletonSensor::~SkeletonSensor()
{
    context_.Shutdown();
}

int SkeletonSensor::initialize()
{
    context_.Init();

    XnStatus rc = XN_STATUS_OK;

    // create depth and user generators
    rc = depthG_.Create(context_);

    if(CHECK_RC(rc, "Create depth generator") == -1)
        return -1;

    rc = userG_.Create(context_);

    if(CHECK_RC(rc, "Create user generator") == -1)
        return -1;
        
    rc = imageG_.Create(context_);
    
     if(CHECK_RC(rc, "Create image generator") == -1)
        return -1;

    XnMapOutputMode mapMode;
    depthG_.GetMapOutputMode(mapMode);

    // for now, make output map VGA resolution at 30 FPS
    mapMode.nXRes = XN_VGA_X_RES;
    mapMode.nYRes = XN_VGA_Y_RES;
    mapMode.nFPS  = 30;

    depthG_.SetMapOutputMode(mapMode);
    imageG_.SetMapOutputMode(mapMode);

    // turn on device mirroring
    if(depthG_.IsCapabilitySupported("Mirror") == true)
    {
        rc = depthG_.GetMirrorCap().SetMirror(true);
        CHECK_RC(rc, "Setting Image Mirroring on depthG");
    }
    
    // turn on device mirroring
    if(imageG_.IsCapabilitySupported("Mirror") == true)
    {
        rc = imageG_.GetMirrorCap().SetMirror(true);
        CHECK_RC(rc, "Setting Image Mirroring on imageG");
    }

    // make sure the user points are reported from the POV of the depth generator
    userG_.GetAlternativeViewPointCap().SetViewPoint(depthG_);
    depthG_.GetAlternativeViewPointCap().SetViewPoint(imageG_);

    // set smoothing factor
    userG_.GetSkeletonCap().SetSmoothing(0.9);

    // start data streams
    context_.StartGeneratingAll();

    // setup callbacks
    setCalibrationPoseCallbacks();

    return 0;
}

void SkeletonSensor::waitForDeviceUpdateOnUser()
{
    context_.WaitOneUpdateAll(userG_);
    updateTrackedUsers();
}

void SkeletonSensor::updateTrackedUsers()
{
    XnUserID users[64];
    XnUInt16 nUsers = userG_.GetNumberOfUsers();

    trackedUsers_.clear();

    userG_.GetUsers(users, nUsers);

    for(int i = 0; i < nUsers; i++)
    {
        if(userG_.GetSkeletonCap().IsTracking(users[i]))
        {
            trackedUsers_.push_back(users[i]);
        }
    }
}

bool SkeletonSensor::isTracking(const unsigned int uid)
{
    return userG_.GetSkeletonCap().IsTracking(uid);
}

Skeleton SkeletonSensor::getSkeleton(const unsigned int uid)
{
    Skeleton result;

    // not tracking user
    if(!userG_.GetSkeletonCap().IsTracking(uid))
        return result;

    // Array of available joints
    const unsigned int nJoints = 15;
    XnSkeletonJoint joints[nJoints] = 
    {   XN_SKEL_HEAD,
        XN_SKEL_NECK,
        XN_SKEL_RIGHT_SHOULDER,
        XN_SKEL_LEFT_SHOULDER,
        XN_SKEL_RIGHT_ELBOW,
        XN_SKEL_LEFT_ELBOW,
        XN_SKEL_RIGHT_HAND,
        XN_SKEL_LEFT_HAND,
        XN_SKEL_RIGHT_HIP,
        XN_SKEL_LEFT_HIP,
        XN_SKEL_RIGHT_KNEE,
        XN_SKEL_LEFT_KNEE,
        XN_SKEL_RIGHT_FOOT,
        XN_SKEL_LEFT_FOOT,
        XN_SKEL_TORSO 
    };

    // holds the joint position components
    XnSkeletonJointPosition positions[nJoints];

    for (unsigned int i = 0; i < nJoints; i++)
    {
        userG_.GetSkeletonCap().GetSkeletonJointPosition(uid, joints[i], *(positions+i));
    }

    SkeletonPoint points[15];
    convertXnJointsToPoints(positions, points, nJoints);

    result.head              = points[0];
    result.neck              = points[1];
    result.rightShoulder     = points[2];
    result.leftShoulder      = points[3];
    result.rightElbow        = points[4];
    result.leftElbow         = points[5];
    result.rightHand         = points[6];
    result.leftHand          = points[7];
    result.rightHip          = points[8];
    result.leftHip           = points[9];
    result.rightKnee         = points[10];
    result.leftKnee          = points[11];
    result.rightFoot         = points[12];
    result.leftFoot          = points[13];
    result.torso             = points[14];

    return result;
}

std::vector<Skeleton> SkeletonSensor::getSkeletons()
{
    std::vector<Skeleton> skeletons;

    for(unsigned int i = 0; i < getNumTrackedUsers(); i++)
    {
        Skeleton s = getSkeleton(trackedUsers_[i]);
        skeletons.push_back(s);
    }

    return skeletons;
}

unsigned int SkeletonSensor::getNumTrackedUsers()
{
    return trackedUsers_.size();
}

unsigned int SkeletonSensor::getUID(const unsigned int index)
{
    return trackedUsers_[index];
}

void SkeletonSensor::setPointModeToProjective()
{
    pointModeProjective_ = true;
}

void SkeletonSensor::setPointModeToReal()
{
    pointModeProjective_ = false;
}

const XnDepthPixel* SkeletonSensor::getDepthData()
{
    return depthG_.GetDepthMap();
}

const XnUInt8* SkeletonSensor::getImageData()
{
    return imageG_.GetImageMap();
}

const XnLabel* SkeletonSensor::getLabels()
{
    xn::SceneMetaData sceneMD;
    
    userG_.GetUserPixels(0, sceneMD);
    
    return sceneMD.Data();
}

int SkeletonSensor::setCalibrationPoseCallbacks()
{
    XnCallbackHandle hUserCallbacks, hCalibrationStart, hCalibrationComplete;

    userG_.RegisterUserCallbacks(newUserCallback, lostUserCallback, this, hUserCallbacks);
    userG_.GetSkeletonCap().RegisterToCalibrationStart(calibrationStartCallback, this, hCalibrationStart);
    userG_.GetSkeletonCap().RegisterToCalibrationComplete(calibrationCompleteCallback, this, hCalibrationComplete);

    // turn on tracking of all joints
    userG_.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);

    return 0;
}

void SkeletonSensor::convertXnJointsToPoints(XnSkeletonJointPosition* const joints, SkeletonPoint* const points, unsigned int numPoints)
{
    XnPoint3D xpt;

    for(unsigned int i = 0; i < numPoints; i++)
    {
        xpt = joints[i].position;

        if(pointModeProjective_)
            depthG_.ConvertRealWorldToProjective(1, &xpt, &xpt);

        points[i].confidence = joints[i].fConfidence;
        points[i].x = xpt.X;
        points[i].y = xpt.Y;
        points[i].z = xpt.Z;
    }
}

void XN_CALLBACK_TYPE SkeletonSensor::newUserCallback(xn::UserGenerator& generator, XnUserID nId, void* pCookie)
{
    put_flog(LOG_DEBUG, "New user %d, auto-calibrating", nId);

    SkeletonSensor* sensor = (SkeletonSensor*) pCookie;
    sensor->userG_.GetSkeletonCap().RequestCalibration(nId, true);
}

void XN_CALLBACK_TYPE SkeletonSensor::lostUserCallback(xn::UserGenerator& generator, XnUserID nId, void* pCookie)
{
    put_flog(LOG_DEBUG, "Lost user %d", nId);
}

void XN_CALLBACK_TYPE SkeletonSensor::calibrationStartCallback(xn::SkeletonCapability& capability, XnUserID nId, void* pCookie)
{
    put_flog(LOG_DEBUG, "Calibration started for user %d", nId);
}

void XN_CALLBACK_TYPE SkeletonSensor::calibrationCompleteCallback(xn::SkeletonCapability& capability, XnUserID nId, XnCalibrationStatus eStatus, void* pCookie)
{
    SkeletonSensor* sensor = (SkeletonSensor*) pCookie;

    if(eStatus == XN_CALIBRATION_STATUS_OK)
    {
        put_flog(LOG_DEBUG, "Calibration completed: start tracking user %d", nId);

        sensor->userG_.GetSkeletonCap().StartTracking(nId);
    }
    else
    {
        put_flog(LOG_DEBUG, "Calibration failed for user %d", nId);

        sensor->userG_.GetSkeletonCap().RequestCalibration(nId, true);
    }
}

void XN_CALLBACK_TYPE SkeletonSensor::poseDetectedCallback(xn::PoseDetectionCapability& capability, const XnChar* strPose, XnUserID nId, void* pCookie)
{
    put_flog(LOG_DEBUG, "Pose detected for user %d", nId);

    SkeletonSensor* sensor = (SkeletonSensor*) pCookie;

    sensor->userG_.GetPoseDetectionCap().StopPoseDetection(nId);
    sensor->userG_.GetSkeletonCap().RequestCalibration(nId, true);
}
