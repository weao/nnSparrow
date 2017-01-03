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
// at: type of activation function (SIGMOID, TANH, RECTIFIER, SOFTPLUS, ORIGINAL)
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
// w: filter width
// h: filter height
// nm: number of feature maps
// at: type of activation function
```
```
nnLayer* addPWSConvLayer(nnLayer* pl, int w, int h, int sw, int sh, int nm, int stpx = 1, int stpy = 1, int at = SIGMOID);
// Add a partial weights sharing convolution layer.
// pl: previous layer to connect
// w: filter width
// h: filter height
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
// w: filter width
// h: filter height
```
```
nnLayer* addAvgPoolingLayer(nnLayer* pl, int w, int h);
// Add an average pooling layer.
// pl: previous layer to connect
// w: filter width
// h: filter height
```
**2. Train & Predict**
```
bool train(std::vector<std::vector<double> > &samples, std::vector<int> &labels);
// samples: training data samples, each sample is presented as a vector.
// labels: training labels corresponded to the data samples.
```
```
bool predict(std::vector<double> &sample, int &label, double *ovec=NULL);
// sample: input data sample
// label: outputed label of the input data sample.
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
// Get the number of layers in the model.
```
```
double getAvgError();
// Get the average training error.
```
```
clock_t getRunTime();
// Get the total training time.
```
```
void reset();
// Reset the model.
```
```
void setLearningRate(double a);
```
```
void setLearningDecayRate(double a);
```
```
void setWeightDecay(double d);
```
```
void setMomentum(double a);
```
```
void setTrainBatchCount(int n);
```
```
void setErrorBound(double err);
// Set the maximum training error to terminate the optimization.
```
```
void setEpochCount(int n);
```
```
void setCallbackFunction(void (*f)(void*));
// Set a user defined callback function. The function is called after each epoch.
```

