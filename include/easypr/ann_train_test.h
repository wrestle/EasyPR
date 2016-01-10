#ifndef ANN_TRAIN_TEST_INCLUDE_H
#define ANN_TRAIN_TEST_INCLUDE_H

#include <opencv2/opencv.hpp>
void annTrain(cv::Mat TrainData, cv::Mat classes, int nNeruns);
int saveTrainData();
void saveModel(int _predictsize, int _neurons);
void start();


#endif