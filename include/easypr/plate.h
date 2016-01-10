//////////////////////////////////////////////////////////////////////////
// Name:	    plate Header
// Version:		1.0
// Date:	    2015-03-12
// Author:	    liuruoze
// Copyright:   liuruoze
// Desciption:
// An abstract class for car plate.
//////////////////////////////////////////////////////////////////////////
#ifndef __PLATE_H__
#define __PLATE_H__

#include "core_func.h"

/*! \namespace easypr
Namespace where all the C++ EasyPR functionality resides
*/
namespace easypr {

class CPlate {
 public:
  bool bColored;

  //! ���캯��
  CPlate();

  //! �������ȡ����
  inline void setPlateMat(cv::Mat param) { m_plateMat = param; }
  inline cv::Mat getPlateMat() const { return m_plateMat; }

  inline void setPlatePos(cv::RotatedRect param) { m_platePos = param; }
  inline cv::RotatedRect getPlatePos() const { return m_platePos; }

  inline void setPlateStr(cv::String param) { m_plateStr = param; }
  inline cv::String getPlateStr() const { return m_plateStr; }

  inline void setPlateLocateType(LocateType param) { m_locateType = param; }
  inline LocateType getPlateLocateType() const { return m_locateType; }

 private:
  //! ���Ƶ�ͼ��
  cv::Mat m_plateMat;

  //! ������ԭͼ��λ��
  cv::RotatedRect m_platePos;

  //! �����ַ���
  cv::String m_plateStr;

  //! ���ƶ�λ�ķ���
  LocateType m_locateType;
};

} /*! \namespace easypr*/

#endif /* endif __PLATE_H__ */