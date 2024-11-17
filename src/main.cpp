#include <iostream>
#include <Eigen/Dense>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

int main() {
    // Test Eigen
    Eigen::MatrixXd mat(2, 2);
    mat(0, 0) = 3;
    mat(1, 0) = 2.5;
    mat(0, 1) = -1;
    mat(1, 1) = mat(1, 0) + mat(0, 1);
    std::cout << "Here is the matrix mat:\n" << mat << std::endl;

#ifdef USE_OPENCV
    // Test OpenCV
    cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::circle(image, cv::Point(50, 50), 40, cv::Scalar(255, 0, 0), -1);
    cv::imshow("Test Image", image);
    cv::waitKey(0);
#endif

    return 0;
}
