nnSparrow
---------------------------

The code is tested on Ubuntu 14.04 LTS

> **How to Use:**
>- 1. RUN example: Just type 'make' under command line, then type './example'.
>- 2. IMPORT nnSparrow: Just include all *.hpp files into your project.

--------------------------------

> **Features:**
>- 1. No required external libraries.
>- 2. Fast on CPU.
>- 3. Easy to use.

--------------------------------

Breif Summary of nnSparrow's member functions:

**1. Adding Layers**
```
nnLayer* addInputLayer(int w, int h, int ch);
// Add an input layer (maximum 2D). 
// w: input width
// h: input height
// ch: number of channels
```
```
nnLayer* addFullLayer(nnLayer* pl, int n, int at = SIGMOID);
// Add a fully connected layer. 
// pl: previous layer to connect
// n: number of units
// at: type of activation function
```
```
nnLayer *addSoftmaxLayer(nnLayer* pl, int n);
// Add a softmax layer. 
// pl: previous layer to connect
// n: number of units
```
```
nnLayer* addFWSConvLayer(nnLayer* pl, int w, int h, int nm, int at = SIGMOID);
// Add a full weights sharing convolution layer.
// pl: previous layer to connect
// w: input width
// h: input height
// nm: number of feature maps
// at: type of activation function
```
```
nnLayer* addPWSConvLayer(nnLayer* pl, int w, int h, int sw, int sh, int nm, int stpx = 1, int stpy = 1, int at = SIGMOID);
// Add a partial weights sharing convolution layer.
// pl: previous layer to connect
// w: input width
// h: input height
// nm: number of feature maps
// sw: section width
// hw: section height
// stpx: stride x
// stpy: stride y
// at: type of activation function
```
```
nnLayer* addMaxPoolingLayer(nnLayer* pl, int w, int h);
// Add a max pooling layer.
// pl: previous layer to connect
// w: input width
// h: input height
```
```
nnLayer* addAvgPoolingLayer(nnLayer* pl, int w, int h);
// Add an average pooling layer.
// pl: previous layer to connect
// w: input width
// h: input height
```
**2. Train & Predict**
```
bool train(std::vector<std::vector<double> > &samples, std::vector<int> &labels);
```
```
bool predict(std::vector<double> &sample, int &label, double *ovec=NULL);
```
```
void load(const char *path);
```
```
void save(const char *path);
```
**3. Configuration**
```
int getLayerCount();
// Get the number of layers.
```
```
double getAvgError();
// Get the average training error.
```
```
clock_t getRunTime();
// Get the running time.
```
```
void reset();
// Reset the model.
```
```
void setLearningRate(double a);
// Set the learning rate
```
```
void setLearningDecayRate(double a);
// Set the learning decay rate.
```
```
void setWeightDecay(double d);
// Set the weight decay parameter.
```
```
void setMomentum(double a);
// Set the momentum parameter.
```
```
void setTrainBatchCount(int n);
// Set the training batch count.
```
```
void setErrorBound(double err);
// Set the error bound.
```
```
void setEpochCount(int n);
// Set the training epoch count.
```
```
void setCallbackFunction(void (*f)(void*));
// Set a user defined callback function. The function is called after each epoch.
```

