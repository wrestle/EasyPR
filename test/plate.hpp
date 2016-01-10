#ifndef EASYPR_PLATE_HPP
#define EASYPR_PLATE_HPP

namespace easypr {

namespace demo {

//using namespace cv;
using namespace std;

int test_plate_locate() {
  cout << "test_plate_locate" << endl;

  const string file = "C:/git/EasyPR/EasyPR/resources/image/test.jpg";

  cv::Mat src = cv::imread(file);
  if (src.data == NULL)
	  std::cerr << "Data is Null" << std::endl;
  
  //TODO：原plateLocate需要被替换

  vector<cv::Mat> resultVec;
  CPlateLocate plate;
  plate.setDebug(1);
  plate.setLifemode(true);

  int result = plate.plateLocate(src, resultVec);
  if (result == 0) {
    size_t num = resultVec.size();
    for (size_t j = 0; j < num; j++) {
      cv::Mat resultMat = resultVec[j];
      imshow("plate_locate", resultMat);
      cv::waitKey(0);
    }
    cv::destroyWindow("plate_locate");
  }
  std::cout << "test_plate_locate end" << std::endl;
  return result;
}

int test_plate_judge() {
  cout << "test_plate_judge" << endl;

  cv::Mat src = cv::imread("C:/git/EasyPR/EasyPR/resources/image/plate_judge.jpg");

  //可能是车牌的图块集合
  vector<cv::Mat> matVec;

  //经过SVM判断后得到的图块集合
  vector<cv::Mat> resultVec;

  CPlateLocate lo;
  lo.setDebug(1);
  lo.setLifemode(true);

  int resultLo = lo.plateLocate(src, matVec);

  if (0 != resultLo)
    return -1;

  cout << "plate_locate_img" << endl;
  size_t num = matVec.size();
  for (size_t j = 0; j < num; j++) {
    cv::Mat resultMat = matVec[j];
    cv::imshow("plate_judge", resultMat);
    cv::waitKey(0);
	//std::cout << "Debug <<<<<<< " << j << "Times Occur!" << std::endl;
  }
  
  cv::destroyWindow("plate_judge");
  
  CPlateJudge ju;
 // std::cout << "Debug <<<< create CPlateJudge" << std::endl;

  ju.LoadModel("C:/git/EasyPR/EasyPR/resources/model/svm.xml");
  int resultJu = ju.plateJudge(matVec, resultVec);

 // std::cerr << "Debug <<<< Iam here after plateJudge" << std::endl;

  if (0 != resultJu)
    return -1;

  cout << "plate_judge_img" << endl;
  num = resultVec.size();
  for (size_t j = 0; j < num; j++) {
    cv::Mat resultMat = resultVec[j];
	cv::imshow("plate_judge", resultMat);
	cv::waitKey(0);
  }
  cv::destroyWindow("plate_judge");
  std::cout << "test_plate_judge end!" << endl;
  return resultJu;
}

int test_plate_detect() {
  cout << "test_plate_detect" << endl;

  cv::Mat src = cv::imread("C:/git/EasyPR/EasyPR/resources/image/plate_detect.jpg");

  vector<CPlate> resultVec;
  CPlateDetect pd;
  pd.setPDLifemode(true);

  int result = pd.plateDetect(src, resultVec);
  if (result == 0) {
    size_t num = resultVec.size();
    for (size_t j = 0; j < num; j++) {
      CPlate resultMat = resultVec[j];

	  cv::imshow("plate_detect", resultMat.getPlateMat());
	  cv::waitKey(0);
    }
	cv::destroyWindow("plate_detect");
  }
  std::cout << "test_plate_detect end!" << endl;
  return result;
}

int test_plate_recognize() {
  cout << "test_plate_recognize" << endl;

  cv::Mat src = cv::imread("C:/git/EasyPR/EasyPR/resources/image/plate_recognize.jpg");

  CPlateRecognize pr;
  pr.LoadANN("C:/git/EasyPR/EasyPR/resources/model/ann.xml");
  pr.LoadSVM("C:/git/EasyPR/EasyPR/resources/model/svm.xml");

  pr.setLifemode(true);
  pr.setDebug(true);

  vector<string> plateVec;

  int result = pr.plateRecognize(src, plateVec);
  if (result == 0) {
    size_t num = plateVec.size();
    for (size_t j = 0; j < num; j++) {
      cout << "plateRecognize: " << plateVec[j] << endl;
    }
  }

  if (result != 0)
    cout << "result:" << result << endl;
  cout << "test_plate_recognise end!" << endl;
  return result;
}

}

}

#endif //EASYPR_PLATE_HPP
