#include "../../include/easypr/plate_recognize.h"

/*! \namespace easypr
    Namespace where all the C++ EasyPR functionality resides
*/
namespace easypr {

CPlateRecognize::CPlateRecognize() {
  // cout << "CPlateRecognize" << endl;
  // m_plateDetect= new CPlateDetect();
  // m_charsRecognise = new CCharsRecognise();
}

// !����ʶ��ģ��
int CPlateRecognize::plateRecognize(cv::Mat src, std::vector<string> &licenseVec,int index)
{
  // ���Ʒ��鼯��
  vector<CPlate> plateVec;

  // ������ȶ�λ��ʹ����ɫ��Ϣ�����Sobel
  int resultPD = plateDetect(src, plateVec, getPDDebug(), 0);

  if (resultPD == 0) {
    int num = plateVec.size();
    int index = 0;

    //����ʶ��ÿ�������ڵķ���
    for (int j = 0; j < num; j++) {
      CPlate item = plateVec[j];
      cv::Mat plate = item.getPlateMat();

      //��ȡ������ɫ
      string plateType = getPlateColor(plate);

      //��ȡ���ƺ�
      cv::String plateIdentify = "";
	  int resultCR = charsRecognise(plate, plateIdentify);
      if (resultCR == 0) {
        string license = plateType + ":" + plateIdentify;
        licenseVec.push_back(license);
      }
    }
    //����ʶ����̵��˽���

    //�����Debugģʽ������Ҫ����λ��ͼƬ��ʾ��ԭͼ���Ͻ�
    if (getPDDebug() == true) {
      cv::Mat result;
      src.copyTo(result);

      for (int j = 0; j < num; j++) {
        CPlate item = plateVec[j];
        cv::Mat plate = item.getPlateMat();

        int height = 36;
        int width = 136;
        if (height * index + height < result.rows) {
          cv::Mat imageRoi = result(cv::Rect(0, 0 + height * index, width, height));
          addWeighted(imageRoi, 0, plate, 1, 0, imageRoi);
        }
        index++;

        cv::RotatedRect minRect = item.getPlatePos();
        cv::Point2f rect_points[4];
        minRect.points(rect_points);

        cv::Scalar lineColor = cv::Scalar(255, 255, 255);

        if (item.getPlateLocateType() == SOBEL) lineColor = cv::Scalar(255, 0, 0);

        if (item.getPlateLocateType() == COLOR) lineColor = cv::Scalar(0, 255, 0);

        for (int j = 0; j < 4; j++)
          line(result, rect_points[j], rect_points[(j + 1) % 4], lineColor, 2,
               8);
      }

      //��ʾ��λ���ͼƬ
      showResult(result);
    }
  }

  return resultPD;
}

} /*! \namespace easypr*/
