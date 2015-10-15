#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <algorithm>
#include "mnist_parser.h"
#include "nnSparrow/nnSparrow.hpp"
using namespace std;


void loadTrain(vector<vector<double> > &data, vector<int> &label) {

  vector<size_t> mnist_labels;
  parse_mnist_images("./testcase/train-images.idx3-ubyte", &data);
  parse_mnist_labels("./testcase/train-labels.idx1-ubyte", &mnist_labels);

  int mx = *max_element(mnist_labels.begin(), mnist_labels.end());
  for(int i=0;i<mnist_labels.size();i++) {
    label.push_back(mnist_labels[i]);
  }
}

void loadTest(vector<vector<double> > &data, vector<int> &label) {

  vector<size_t> mnist_labels;
  parse_mnist_images("./testcase/t10k-images.idx3-ubyte", &data);
  parse_mnist_labels("./testcase/t10k-labels.idx1-ubyte", &mnist_labels);

  int mx = *max_element(mnist_labels.begin(), mnist_labels.end());
  for(int i=0;i<mnist_labels.size();i++) {
    label.push_back(mnist_labels[i]);
  }
}

vector<vector<double> > train_data;
vector<int> train_label;
vector<vector<double> > test_data;
vector<int> test_label;

void testResult(void *param) {

  nnSparrow *nn = (nnSparrow*)param;

  printf("\nTime consumption: %.2lfs\n", double(clock()-nn->getRunTime())/CLOCKS_PER_SEC);
  int num = 0, cnum = 0;
  int ret = 0;

  for(int i=0;i<test_data.size();i++) {
    if(nn->predict(test_data[i], ret)) {
      num++;
      if(test_label[i] == ret) {
        cnum++;
      }
    }
  }
  double rate = double(cnum)/num;
  printf("Accuracy: %.2lf%%\n", rate*100);

};


int main()
{
  //srand((unsigned int) time(0));

  nnSparrow nn;

  int width, height;
  loadTrain(train_data, train_label);
  loadTest(test_data, test_label);

  cout<<"Loading finished."<<endl;

  train_data.resize(train_data.size());
  train_label.resize(train_label.size());
  test_data.resize(test_data.size());
  test_label.resize(test_label.size());


  //input dimension
  int idim = train_data[0].size();

  //output dimension
  int odim = *max_element(train_label.begin(), train_label.end())+1;


  nn.setEpochCount(20);

  //set callback funtion being called everytime when an epoch finished
  nn.setCallbackFunction(testResult);

  //load trained network
  //nn.load("weights.txt");

  nnLayer *pl = NULL;

  //add input layer with 32 X 32 size and 1 channel
  pl = nn.addInputLayer(32, 32, 1);

  //nn.addPWSConvLayer(5, 5, 5, 5, 6);

  //add a full wights sharing convolutional layer with 5 X 5 filter size and 6 feature maps
  pl = nn.addFWSConvLayer(pl, 5, 5, 6);

  //add a maxpooling layer with 2 X 2 filter size
  pl = nn.addMaxPoolingLayer(pl, 2, 2);

  //add a full connected layer with 120 units output, using tanh activation function.
  pl = nn.addFullLayer(pl, 120, TANH);

  //add a softmax layer(output layer) with #odim units output
  nn.addSoftmaxLayer(pl, odim);


  cout<<"Start training..."<<endl;
  if(nn.train(train_data, train_label)) {
    cout<<"Training finished!"<<endl;
  }

  testResult(&nn);

  //save network
  //nn.save("weights.txt");


  return 0;
}
