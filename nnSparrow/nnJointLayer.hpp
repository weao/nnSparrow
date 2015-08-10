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
#include <vector>

#ifndef __NN_JOINT_LAYER__

#define __NN_JOINT_LAYER__

class nnJointLayer : public nnLayer {

protected:

	std::vector<nnLayer*> _children;

public:
	nnJointLayer(std::vector<nnLayer*> &ch) :	nnLayer(NULL, NULL) {

		int tot = 0;

		int sz = ch.size();
		for(int i=0;i<sz;i++) {
			_children.push_back(ch[i]);
			tot += ch[i]->getTotalUnitCount();
		}

		this->_unit_count = tot;
		this->_prev_unit_count = tot;
		this->_width = tot;
		this->_height = 1;
		this->_map_num = 1;

		_u_a = NULL;
		_u_delta = NULL;
	}
	nnJointLayer() : nnLayer(NULL, NULL) {

		this->_unit_count = 0;
		this->_prev_unit_count = 0;
		this->_width = 0;
		this->_height = 1;
		this->_map_num = 1;

		_u_a = NULL;
		_u_delta = NULL;

	}

	~nnJointLayer() {

	}


	void init() {

		clear();
		int n = this->_unit_count;
		if(n > 0) {
			_u_a = new double[n]; //Eigen::MatrixXd::Zero(n,1);
			memset(_u_a, 0, n*sizeof(double));

			_u_delta = new double[n]; //Eigen::MatrixXd::Zero(n,1);
			memset(_u_delta, 0, n*sizeof(double));
		}

	}

	void addLayer(nnLayer *l) {

		_children.push_back(l);
	}
	void join() {
		int tot = 0;
		int sz = _children.size();
		for(int i=0;i<sz;i++) {
			tot += _children[i]->getTotalUnitCount();
		}
		this->_unit_count = this->_prev_unit_count = tot;
	}

	void forward() {

		int sh = 0;
		for(int i=0;i<_children.size();i++) {
			double *pa = _children[i]->getActivation();
			int n = _children[i]->getTotalUnitCount();
			memcpy(_u_a+sh, pa, sizeof(double)*n);
			sh += n;
		}

	}
	void backpropagation(double mu) {

		int sh = 0;
		for(int i=0;i<_children.size();i++) {
			double *pdt = _children[i]->getDelta();
			int n = _children[i]->getTotalUnitCount();
			memcpy(pdt, _u_delta+sh, sizeof(double)*n);
			sh += n;
		}

	}
	void updateParameters(int m, double alpha, double lambda) {



	}
	void updateDelta() {


	}

	int getTotalUnitCount() {
		return _unit_count;
	}

	void clear() {
		if(_u_a) {
			delete [] _u_a;
			_u_a = NULL;
		}
		if(_u_delta) {
			delete [] _u_delta;
			_u_delta = NULL;
		}
	}

	void write(std::ofstream &fout) {


	}
	void read(std::ifstream &fin) {


	}

};


#endif
