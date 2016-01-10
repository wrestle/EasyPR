#include "../../include/easypr/chars_recognise.h"

/*! \namespace easypr
Namespace where all the C++ EasyPR functionality resides
*/
namespace easypr {

CCharsRecognise::CCharsRecognise() {
  // cout << "CCharsRecognise" << endl;
  m_charsSegment = new CCharsSegment();
  m_charsIdentify = new CCharsIdentify();
}

void CCharsRecognise::LoadANN(string s) {
  m_charsIdentify->LoadModel(s);
}

string CCharsRecognise::charsRecognise(cv::Mat plate) {
  return m_charsIdentify->charsIdentify(plate);
}
int CCharsRecognise::charsRecognise(cv::Mat plate, cv::String & plateLicense) {
	using std::vector;
	using std::string;
	//�����ַ����鼯��

  vector<cv::Mat> matVec;

  string plateIdentify = "";

  int result = m_charsSegment->charsSegment(plate, matVec);
  if (result == 0) {
    int num = matVec.size();
    for (int j = 0; j < num; j++) {
      cv::Mat charMat = matVec[j];
      bool isChinses = false;
      bool isSpeci = false;

      //Ĭ���׸��ַ����������ַ�
      if (j == 0) isChinses = true;
      if (j == 1) isSpeci = true;

      string charcater =
          m_charsIdentify->charsIdentify(charMat, isChinses, isSpeci);

      plateIdentify = plateIdentify + charcater;
    }
  }

  plateLicense = plateIdentify;

  if (plateLicense.size() < 7) {
    return -1;
  }

  return result;
}

} /*! \namespace easypr*/
