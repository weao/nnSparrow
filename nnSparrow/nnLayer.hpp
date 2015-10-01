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

#include <cmath>
#include <fstream>
#include <memory.h>
#include <cstdlib>
#include <cstdio>
#include <cfloat>

#include "nnActivation.hpp"

#ifndef __NN_LAYER__
#define __NN_LAYER__
#define MIN(a,b) ((a)<(b)?(a):(b))

class nnLayer {

protected:
	nnLayer* _prev;
	nnLayer* _next;

	double* _u_a;
	double* _u_delta;
	double* _u_W;
	double* _u_b;

	int _unit_count;
	int _prev_unit_count;
	int _width;
	int _height;
	int _map_num;
	int _layer_type;

	activation _act_f;
	activation _d_act_f;
public:
	enum LAYER_TYPE {
		DEFAULT_LAYER = 1,
		INPUT_LAYER ,
		MAX_POOLING_LAYER,
		AVG_POOLING_LAYER,
		FWS_CONV_LAYER,
		FULL_LAYER,
		SOFTMAX_LAYER
	};

	nnLayer(nnLayer *prev, nnLayer *next) {

		this->_prev = prev;
		this->_next = next;

		_width = 0;
		_height = 0;
		_unit_count = 0;
		_prev_unit_count = 0;
		_map_num = 0;
		_layer_type = DEFAULT_LAYER;

		_u_a = NULL;
		_u_delta = NULL;
		_u_W = NULL;
		_u_b = NULL;
	}
	~nnLayer() {

		if(_u_a)
			delete [] _u_a;
		if(_u_delta)
			delete [] _u_delta;
		if(_u_W)
			delete [] _u_W;
		if(_u_b)
			delete [] _u_b;
	}

	virtual void write(std::ofstream &fout) = 0;
	virtual void read(std::ifstream &fin) = 0;

	nnLayer *getPrevLayer() {
		return _prev;
	}
	nnLayer *getNextLayer() {
		return _next;
	}
	void setNextLayer(nnLayer *l) {
		_next = l;
	}
	void setPrevLayer(nnLayer *l) {
		_prev = l;
	}

	int getWidth() {
		return _width;
	}
	int getHeight() {
		return _height;
	}
	int getMapNum() {
		return _map_num;
	}

	int getUnitCount() {
		return _unit_count;
	}

	double* getActivation() {
		return _u_a;
	}


	double* getDelta() {
		return _u_delta;
	}



	virtual void init() = 0;
	virtual void updateDelta() = 0;
	virtual void backpropagation(double) = 0;
	virtual void forward() = 0;
	virtual void updateParameters(int,double,double) = 0;
	virtual int getTotalUnitCount() = 0;

	void setDelta(double *a, int n) {

		memcpy(_u_delta, a, sizeof(double)*n);
	}
	double* getWeights() {
		return _u_W;
	}
};


#endif
