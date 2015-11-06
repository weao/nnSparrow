/*
    Copyright (c) 2015, Weihao Cheng
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "nnLayer.hpp"
#include "nnInputLayer.hpp"
#include "nnFLayer.hpp"
#include "nnJointLayer.hpp"
#include "nnFWSConvLayer.hpp"
#include "nnMaxPoolingLayer.hpp"
#include "nnAvgPoolingLayer.hpp"
#include "nnPWSConvLayer.hpp"
#include "nnSoftmaxLayer.hpp"
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cfloat>
#include <ctime>

#ifndef __NN_SPARROW__
#define __NN_SPARROW__


class nnSparrow {


protected:

	void (*_call_back)(void*);
	std::vector<nnLayer*> _layers;
	std::vector<nnInputLayer*> _inputlayers;
	double _learning_decay_rate;
	double _learning_rate;
	double _weight_decay_parameter;
	double _error_bound;
	double _avg_error;
	int _epoch_count;
	double _momentum;
	int _train_batch_count;
	clock_t _run_time;
	bool _ready;

public:
	nnSparrow() {
		_momentum = 0.9;
		_learning_rate = 0.01;
		_weight_decay_parameter = 0.0001;
		_error_bound = 0.00001;
		_epoch_count = 20;
		_train_batch_count = 10;
		_avg_error = 0;
		_ready = false;
		_call_back = NULL;
		_learning_decay_rate = 0.9;
		_run_time = clock();

	}

	~nnSparrow() {
		reset();
	}

	clock_t getRunTime() {
		return _run_time;
	}

	void setCallbackFunction(void (*f)(void*)) {
		this->_call_back = f;
	}
	void setErrorBound(double err) {
		_error_bound = err;
	}
	void setEpochCount(int n) {
		_epoch_count = n;
	}

	void setLearningRate(double a) {
		this->_learning_rate = a;
	}
	void setWeightDecay(double d) {
		this->_weight_decay_parameter = d;
	}
	void setTrainBatchCount(int n) {
		this->_train_batch_count = n;
	}
	void setMomentum(double a) {
		this->_momentum = a;
	}
	void setLearningDecayRate(double a) {
		this->_learning_decay_rate = a;
	}

	double getAvgError() {
		return this->_avg_error;
	}

	void reset() {
		while(!_layers.empty()) {
			delete _layers.back();
			_layers.pop_back();
		}
		while(!_inputlayers.empty()) {
			delete _inputlayers.back();
			_inputlayers.pop_back();
		}
		_ready = false;
	}


	int getLayerCount() {
		return _layers.size();
	}

	void addLayer(nnLayer *l) {

		_layers.push_back(l);
	}
	void save(const char *path) {

		std::ofstream fout(path);
		fout << _momentum << " " << _learning_rate << " " << _weight_decay_parameter << std::endl;
		fout << _layers.size() << std::endl;
		for(int i=0;i<_layers.size();i++) {
			_layers[i]->write(fout);
		}
	}

	void load(const char *path) {

		std::ifstream fin(path);
		fin >> _momentum >> _learning_rate >> _weight_decay_parameter;
		int n = 0;
		fin >> n;
		for(int i=0;i<n;i++) {
			int type;
			fin >> type;
			nnLayer *l;
			switch(type) {
				case nnLayer::INPUT_LAYER:
					l = new nnInputLayer();
					break;
				case nnLayer::FULL_LAYER:
					l = new nnFLayer();
					break;
				case nnLayer::FWS_CONV_LAYER:
					l = new nnFWSConvLayer();
					break;
				case nnLayer::MAX_POOLING_LAYER:
					l = new nnMaxPoolingLayer();
					break;
				case nnLayer::SOFTMAX_LAYER:
					l = new nnSoftmaxLayer();
					break;
				default:
					break;
			}
			_layers.push_back(l);
			if(i > 0) {
				_layers[i]->setPrevLayer(_layers[i-1]);
			}
			l->read(fin);
		}
		_ready = true;
	}

	nnLayer* addInputLayer(int w, int h, int ch) {
		//reset();
		nnInputLayer *l = new nnInputLayer(w, h, ch);
		//_layers.push_back(l);
		_inputlayers.push_back(l);
		return l;
	}

	nnLayer* addFWSConvLayer(nnLayer* pl, int w, int h, int nm, int at = SIGMOID) {
		// if(_layers.empty())
		// 	return NULL;
		//nnLayer *pl = _layers.back();
		nnFWSConvLayer *l = new nnFWSConvLayer(w, h, nm, at, pl, NULL);
		_layers.push_back((nnLayer*)l);
		if(pl) {
			pl->setNextLayer((nnLayer*)l);
		}
		return l;
	}

	nnLayer* addPWSConvLayer(nnLayer* pl, int w, int h, int sw, int sh, int nm, int at = SIGMOID) {
		// if(_layers.empty())
		// 	return NULL;
		//nnLayer *pl = _layers.back();
		nnPWSConvLayer *l = new nnPWSConvLayer(w, h, sw, sh, nm, at, pl);
		_layers.push_back((nnLayer*)l);
		if(pl) {
			pl->setNextLayer((nnLayer*)l);
		}
		return l;
	}

	nnLayer* addJointLayer(std::vector<nnLayer*> &ch) {
		nnJointLayer *l = new nnJointLayer(ch);
		_layers.push_back((nnLayer*)l);
		for(int i=0;i<ch.size();i++) {
			ch[i]->setNextLayer((nnLayer*)l);
		}
		return l;
	}


	nnLayer* addMaxPoolingLayer(nnLayer* pl, int w, int h) {

		// if(_layers.empty())
		// 	return NULL;
		//nnLayer *pl = _layers.back();
		nnMaxPoolingLayer *l = new nnMaxPoolingLayer(w, h, pl, NULL);
		_layers.push_back((nnLayer*)l);
		if(pl) {
			pl->setNextLayer((nnLayer*)l);
		}
		return l;

	}

	nnLayer* addAvgPoolingLayer(nnLayer* pl, int w, int h) {

		// if(_layers.empty())
		// 	return NULL;
		//nnLayer *pl = _layers.back();
		nnAvgPoolingLayer *l = new nnAvgPoolingLayer(w, h, pl, NULL);
		_layers.push_back((nnLayer*)l);
		if(pl) {
			pl->setNextLayer((nnLayer*)l);
		}
		return l;

	}


	nnLayer* addFullLayer(nnLayer* pl, int n, int at = SIGMOID) {

		// if(_layers.empty())
		// 	return NULL;
		//nnLayer *pl = _layers.back();

		nnFLayer *l = new nnFLayer(n, at, pl, NULL);
		_layers.push_back((nnLayer*)l);
		if(pl) {
			pl->setNextLayer((nnLayer*)l);
		}
		return l;
	}

	nnLayer *addSoftmaxLayer(nnLayer* pl, int n) {

			// if(_layers.empty())
			// 	return NULL;
			//nnLayer *pl = _layers.back();

			nnFLayer *l = new nnSoftmaxLayer(n, pl, NULL);
			_layers.push_back((nnLayer*)l);
			if(pl) {
				pl->setNextLayer((nnLayer*)l);
			}
			return l;
	}


	void prepare() {

		for(int i=0;i<_layers.size();i++) {
			_layers[i]->init();
		}
		for(int i=0;i<_inputlayers.size();i++) {
			_inputlayers[i]->init();
		}

	}



	bool train(std::vector<std::vector<double> > &input, std::vector<int> &output) {
		if(_inputlayers.size() < 1 || _layers.size() < 1)
			return false;
		if(input.size() <= 0 || input.size() != output.size())
			return false;
		if(_inputlayers.front()->getTotalUnitCount() != input[0].size())
			return false;
		if(!_ready) {
			prepare();
			_ready = true;
		}

		int sz = _layers.size();

		int len = input.size();
		//int dim = input[0].size();
		//int odim = output[0].size();
		int odim = 0;
		for(int i=0;i<output.size();i++) {
			if(output[i]+1 > odim)
				odim = output[i]+1;
		}
		_avg_error = 100;

		double *ovec = new double[odim];

		//nnInputLayer *input_layer = (nnInputLayer*)_layers.front();
		nnFLayer *output_layer = (nnFLayer*)_layers.back();

		int *rank = new int[len];
		for(int i=0;i<len;i++)
			rank[i] = i;

		double E = 0;
		unsigned long long itr;
		unsigned long long tot = this->_epoch_count * len;
		for(itr = 0; itr < tot; itr++) {

			int idx = itr % len;
			if(idx == 0) {

				//shuffle is important!!!
				for(int i=0;i<len;i++) {
					int j = i + rand() % (len - i);
					std::swap(rank[i], rank[j]);
				}

				if(itr > 0) {
					E /= len;
					if(itr > 0 && fabs(E-_avg_error) < _error_bound) {
							printf("%lf %lf\n", E, _avg_error );
							break;
					}
					_avg_error = E;
					E = 0;
					_learning_rate *= _learning_decay_rate;
					if(this->_call_back)
						this->_call_back(this);
				}
				_run_time = clock();
			}

			//printf("%d\n", idx);
			idx = rank[idx];

			for(int i=0;i<_inputlayers.size();i++)
				_inputlayers[i]->inputSample(&input[idx][0], input[idx].size());

			for(int j=0;j<sz;j++) {
				_layers[j]->forward();
			}


			memset(ovec, 0, sizeof(double)*odim);
			ovec[output[idx]] = 1;

			output_layer->calculateDelta(ovec, odim);
			double *a = output_layer->getActivation();
			for(int i=0;i<odim;i++) {
				double t = a[i] - ovec[i];
				E += fabs(t);
			}

			for(int j=sz-1;j>=0;j--) {
				_layers[j]->backpropagation();
			}

			if(itr % _train_batch_count == 0) {
				for(int j=sz-1;j>=0;j--) {
					_layers[j]->updateParameters(_train_batch_count, _learning_rate, _weight_decay_parameter, _momentum);
				}
			}
		}

		delete [] rank;
		delete [] ovec;

		return true;
	}

	bool predict(std::vector<double> &input, int &output, double *ovec=NULL) {

		if(_inputlayers.size() < 1 || _layers.size() < 1)
			return false;
		int dim = input.size();
		if(_inputlayers.front()->getTotalUnitCount() != dim)
			return false;

		for(int i=0;i<_inputlayers.size();i++)
			_inputlayers[i]->inputSample(&input[0], dim);

		int sz = _layers.size();
		for(int i=0;i<sz;i++)
			_layers[i]->forward();

		output = 0;

		double *af = ((nnFLayer*)_layers.back())->getActivation();
		for(int i=1;i<_layers.back()->getUnitCount();i++) {
			if(af[i] > af[output])
				output = i;
		}

		if(ovec) {
			for(int i=0;i<_layers.back()->getUnitCount();i++)
				ovec[i] = af[i];
		}


		return true;
	}
};

#endif
