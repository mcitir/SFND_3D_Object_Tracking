#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType, double &matcherTime)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    matcherTime = 0.0;
    
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // OpenCV bug fix
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F)
        {
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "FLANN matching done" << endl;
    }
    else
    {
        throw invalid_argument("Invalid matcherType: " + matcherType);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "NN matching time = " << 1000 * t / 1.0 << " ms" << endl;
        matcherTime = 1000 * t / 1.0;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "KNN matching time = " << 1000 * t / 1.0 << " ms" << endl;
        matcherTime = 1000 * t / 1.0;

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }

        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
    else
    {
        throw invalid_argument("Invalid selectorType: " + selectorType);
    }
    
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double &descriptorTime)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {

        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {

        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();        
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
            
        extractor = cv::SIFT::create();
    }
    else 
    {
        throw invalid_argument("Invalid descriptorType: " + descriptorType);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    descriptorTime = 1000 * t / 1.0; // save descriptor time
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, double &detectionTime)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    detectionTime = 1000 * t / 1.0; // save detection time

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the Harris detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, double &detectionTime)
{
    //detectionTime = 0.0; // save detection time

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Non-maximum suppression (NMS) settings
    bool bNMS = true; // perform non-maximum suppression on keypoints
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maximum suppression (NMS)

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;

    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    // Normalize
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // Scale
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Visualize results
    if (bVis)
    {
        // Draw circles around detected corners
        cv::Mat dst_norm_scaled_copy = dst_norm_scaled.clone();
        for (int j = 0; j < dst_norm_scaled_copy.rows; j++)
        {
            for (int i = 0; i < dst_norm_scaled_copy.cols; i++)
            {
                if ((int)dst_norm_scaled_copy.at<uchar>(j, i) > minResponse)
                {
                    cv::circle(dst_norm_scaled_copy, cv::Point(i, j), 5, cv::Scalar(0), 2, 8, 0);
                }
            }
        }

        // Display results
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, dst_norm_scaled_copy);
        cv::waitKey(0);
    }

    // Apply non-maximum suppression (NMS)
    for (size_t j= 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);

            // Apply the minimum threshold for Harris cornerness response
            if (response > minResponse)
            {
                // Only store points above a threshold
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // Perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {
                            // Replace old key point with new one
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }
                if (!bOverlap)
                {
                    // Add new key point
                    keypoints.push_back(newKeyPoint);
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Harris detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    detectionTime = 1000 * t / 1.0; // save detection time 
}
void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis, double &detectionTime)
{
    // select appropriate descriptor
    //cv::Ptr<cv::FeatureDetector> detector;
    //double t = (double)cv::getTickCount();
    if (detectorType.compare("FAST") == 0)
    {
        // Detector parameters
        int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true; // perform non-maximum suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        
        //double t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "FAST detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        detectionTime = 1000 * t / 1.0; // save detection time

        
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        // Detector parameters
        int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel
        int octaves = 3; // detection octaves
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint

        // Detect keypoints
        //double t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create(threshold, octaves, patternScale);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "BRISK detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        detectionTime = 1000 * t / 1.0; // save detection time

    }
    else if (detectorType.compare("ORB") == 0)
    {
        // Detector parameters
        int nfeatures = 500; // maximum number of features to retain
        float scaleFactor = 1.2f; // pyramid decimation ratio, greater than 1
        int nlevels = 8; // number of pyramid levels
        int edgeThreshold = 31; // size of the border where the features are not detected
        int firstLevel = 0; // level of pyramid to put source image to
        int WTA_K = 2; // number of points that produce each element of the oriented BRIEF descriptor
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE; // algorithm to rank features
        int patchSize = 31; // size of the patch used by the oriented BRIEF descriptor
        int fastThreshold = 20; // threshold for the keypoint detector when local non-maximum suppression is used

        // Detect keypoints
        //double t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "ORB detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        detectionTime = 1000 * t / 1.0; // save detection time

    
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        // Detector parameters
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB; // type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT
        int descriptor_size = 0; // size of the descriptor in bits. 0 -> full size
        int descriptor_channels = 3; // number of channels in the descriptor (1, 2, 3)
        float threshold = 0.001f; // detector response threshold to accept point
        int nOctaves = 4; // maximum octave evolution of the image
        int nOctaveLayers = 4; // default number of sublevels per scale level
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER

        // Detect keypoints
        //double t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "AKAZE detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        detectionTime = 1000 * t / 1.0; // save detection time
    
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        // Detector parameters
        int nfeatures = 0; // The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        int nOctaveLayers = 3; // The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution
        double contrastThreshold = 0.04; // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector
        double edgeThreshold = 10; // The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained)
        double sigma = 1.6; // The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number

        // Detect keypoints
        //double t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "SIFT detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        detectionTime = 1000 * t / 1.0; // save detection time

    
    }
    else
    {
        throw invalid_argument(detectorType + " is not a valid detectorType");
    }

    // // perform feature detection
    // detector->detect(img, keypoints);
    // t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // std::cout << detectorType << " detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // detectionTime = 1000 * t / 1.0; // save detection time

    // Visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}