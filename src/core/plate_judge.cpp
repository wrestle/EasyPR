#include "../../include/easypr/plate_judge.h"

/*! \namespace easypr
    Namespace where all the C++ EasyPR functionality resides
*/
namespace easypr {
	using std::vector;
CPlateJudge::CPlateJudge() {
  // std::cout << "CPlateJudge" << std::endl;
  m_path = "C:/git/EasyPR/EasyPR/resources/model/svm.xml"; // std::string
  m_getFeatures = getHistogramFeatures;
  // std::cout << "Before CPlateJudge::LoadModel()" << std::endl;
  LoadModel();
}

void CPlateJudge::LoadModel() {

	if (!svm.empty())
	svm->clear();
	svm = cv::Algorithm::load<cv::ml::SVM>(m_path);
	//2.4.8
	//svm->load<cv::ml::SVM>(m_path, "svm");//, "svm");
}

void CPlateJudge::LoadModel(std::string s) {
	if (!svm.empty())
	svm->clear();
  svm = cv::Algorithm::load<cv::ml::SVM>(s);
}

//! 直方图均衡
cv::Mat CPlateJudge::histeq(cv::Mat in) {
	//cv class
	using cv::Mat;

  Mat out(in.size(), in.type());
  if (in.channels() == 3) {
    Mat hsv;
    vector<Mat> hsvSplit;
    cvtColor(in, hsv, CV_BGR2HSV);
    split(hsv, hsvSplit);
    equalizeHist(hsvSplit[2], hsvSplit[2]);
    merge(hsvSplit, hsv);
    cvtColor(hsv, out, CV_HSV2BGR);
  } else if (in.channels() == 1) {
    equalizeHist(in, out);
  }
  return out;
}

//! 对单幅图像进行SVM判断
int CPlateJudge::plateJudge(const cv::Mat& inMat, int& result) {
	if (m_getFeatures == NULL) return -1;
	//cv class
	using cv::Mat;
  // std::cerr << "Debug <<<< Iam here In plateJudge(Mat&, int&), before m_getFeatures()" << std::endl;
  Mat features;
  m_getFeatures(inMat, features);

  // std::cerr << "Debug <<<< Iam here In plateJudge(Mat&, int&), after m_getFeatures()" << std::endl;

  //通过直方图均衡化后的彩色图进行预测
  Mat p = features.reshape(1, 1);
  // std::cout << "Debug <<<<<<< after reshape() "<< std::endl;
  p.convertTo(p, CV_32FC1);
  // std::cout << "Debug <<<<<<< after convertTo()" << std::endl;
  //Mat resul;
  float response = svm->predict(p); //, resul, cv::ml::StatModel::RAW_OUTPUT);

  // std::cout << "Debug <<<<<<< response = " << response << std::endl;
  // std::cout << "Debug <<<<<<< after predict" << std::endl;
  result = (int)response;
  // std::cout << "Debug <<<<<<< result      = " << result << std::endl;
  return 0;
}

//! 对多幅图像进行SVM判断
int CPlateJudge::plateJudge(const vector<cv::Mat>& inVec, vector<cv::Mat>& resultVec)
{
	//cv class
	using cv::Mat;
	std::cout << "Debug <<<<<<<< Iam in plateJudge" << std::endl;
  int num = inVec.size();
  for (int j = 0; j < num; j++) {
    Mat inMat = inVec[j];

    int response = -1;
    plateJudge(inMat, response);

    if (response == 1) resultVec.push_back(inMat);
  }
  return 0;
}

//! 对多幅车牌进行SVM判断
int CPlateJudge::plateJudge(const vector<CPlate>& inVec,
                            vector<CPlate>& resultVec)
{
	//cv class
	using cv::Mat;
	using cv::Rect_;
	//cv function
	using cv::Size;

  int num = inVec.size();
  for (int j = 0; j < num; j++) {
    CPlate inPlate = inVec[j];
    Mat inMat = inPlate.getPlateMat();

    int response = -1;
    plateJudge(inMat, response);

    if (response == 1)
      resultVec.push_back(inPlate);
    else {
      int w = inMat.cols;
      int h = inMat.rows;
      //再取中间部分判断一次
      Mat tmpmat = inMat(Rect_<double>(w * 0.05, h * 0.1, w * 0.9, h * 0.8));
      Mat tmpDes = inMat.clone();
      resize(tmpmat, tmpDes, Size(inMat.size()));

      plateJudge(tmpDes, response);

      if (response == 1) resultVec.push_back(inPlate);
    }
  }
  return 0;
}

} /*! \namespace easypr*/
