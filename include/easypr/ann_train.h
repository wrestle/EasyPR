#ifndef EASYPR_ANN_TRAIN_H
#define EASYPR_ANN_TRAIN_H

#include <opencv2/opencv.hpp>

namespace easypr {
	static const char * default_ann_char_path = "C:\\git\\EasyPR\\EasyPR\\resources\\train\\data\\chars_recognise_ann\\chars2";
	static const char * default_ann_model_save = "C:\\git\\EasyPR\\EasyPR\\resources\\train\\ann_train.xml";
	
	//static const char * default_svm_char_path = ""
	class AnnTrain {
	public:
		
		explicit AnnTrain(const char* chars_folder = default_ann_char_path, 
			                const char* xml = default_ann_model_save);

		void train();
	private:
		virtual cv::Ptr<cv::ml::TrainData> tdata();

		cv::Ptr<cv::ml::ANN_MLP> ann_;
		const char* ann_xml_;
		const string chars_folder_;
		const string ann_xml_cpy;
	};

	class CharsIdentify {
	public:
		static CharsIdentify* instance();

		std::pair<std::string, std::string> identify(cv::Mat input);

	private:
		CharsIdentify();

		static CharsIdentify* instance_;
		cv::Ptr<cv::ml::ANN_MLP> ann_;
	};
}
#endif