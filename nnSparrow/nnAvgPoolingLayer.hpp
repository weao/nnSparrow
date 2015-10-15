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

#ifndef __NN_AVG_POOLING_LAYER__

#define __NN_AVG_POOLING_LAYER__

class nnAvgPoolingLayer : public nnLayer {

private:
	int _filter_width;
	int _filter_height;
	int* _u_W_idx;

public:

	nnAvgPoolingLayer() : nnLayer(NULL, NULL) {

		_filter_width = 0;
		_filter_height = 0;
			_u_W_idx = NULL;

	}
	nnAvgPoolingLayer(int fw, int fh, nnLayer *prev, nnLayer *next = NULL) : nnLayer(prev, next) {

		this->_filter_width = fw;
		this->_filter_height = fh;

		int pw = prev->getWidth();
		int ph = prev->getHeight();

		this->_width = pw / fw;
		this->_height = ph / fh;

		this->_map_num = _prev->getMapNum();

		if(_width * fw < pw)
			_width++;
		if(_height * fh < ph)
			_height++;

		this->_unit_count = _width * _height;
		this->_prev_unit_count = prev ? prev->getUnitCount() : 0;
		_u_W_idx = NULL;

		this->_layer_type = AVG_POOLING_LAYER;
	}
	~nnAvgPoolingLayer() {

		if(_u_W_idx)
			delete [] _u_W_idx;
	}

	void init() {
		clear();
		int n = _unit_count, np = _prev_unit_count, nm = _map_num;


		_u_a = new double[n*nm];
		memset(_u_a, 0, sizeof(double)*n*nm);

		_u_delta = new double[n*nm];
		memset(_u_delta, 0, sizeof(double)*n*nm);


		_u_W_idx = new int[n*_filter_height*2];
		memset(_u_W_idx, -1, sizeof(int)*n*_filter_height*2);


		mapFilterToWeights();

	}


	void mapFilterToWeights() {

		int pw = _prev->getWidth();
		int ph = _prev->getHeight();
		int fw = _filter_width;
		int fh = _filter_height;
		int n = _unit_count, np = _prev_unit_count;

		int wn = fh * 2;

		int step = 0;
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < fh * pw; j += pw) {
				if(step + j >= pw * ph)
					break;

				int lu = MIN(fw, (step/pw + 1)*pw-step);

				_u_W_idx[i*wn + 2*j/pw + 0] = step + j + 0;
				_u_W_idx[i*wn + 2*j/pw + 1] = step + j + lu;

			}
			if(step % pw + fw >= pw)
				step = (step / pw + fh) * pw;
			else
				step += fw;
		}
	}

	void forward() {

		const int n = _unit_count, np = _prev_unit_count, nm = _map_num;
		const int pw = _prev->getWidth();
		const int fw = _filter_width;
		const int fh = _filter_height;

		double *ppa = _prev->getActivation();
		double *pa = _u_a;

		int wn = fh*2;

		for(int mi = 0; mi < nm; mi++, ppa += np) {

			int *pidx = _u_W_idx;
			for(int i = 0; i < n; i++, pidx += wn, pa++) {
				double sum = 0;
				int sz = 0;
				for(int j = 0; j < wn; j += 2) {
					int st = *(pidx + j + 0);
					if(st < 0)
						break;
					int ed = *(pidx + j + 1);
					for(int k = st; k < ed; k++) {
						sum += *(ppa + k);
						sz ++;
					}
				}
				*pa = sum / sz;
			}
		}
	}

	void backpropagation() {

		double *ppd = _prev->getDelta();
		const int n = _unit_count, np = _prev_unit_count, nm = _map_num;

		if(ppd) {
			memset(ppd, 0, sizeof(double)*_prev->getTotalUnitCount());
			double *pd = _u_delta;

			int wn = _filter_height*2;

			for(int mi = 0; mi < nm; mi++, pd += n, ppd += np) {

				int *pidx = _u_W_idx;
				for(int i = 0; i < n; i++, pidx += wn) {

					double d = *(pd + i);
					for(int j = 0; j < wn; j += 2) {
						int st = *(pidx + j + 0);
						if(st < 0)
							break;
						int ed = *(pidx + j + 1);
						for(int k = st; k < ed; k++) {
							*(ppd + k) = d;
						}
					}
				}
			}

			_prev->updateDelta();
		}
	}

	void updateDelta() {
	}

	void updateParameters(int m, double alpha, double lambda, double mu) {

	}
	int getTotalUnitCount() {
		return _unit_count * _map_num;
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
		if(_u_W_idx) {
			delete [] _u_W_idx;
			_u_W_idx = NULL;
		}
	}
	void write(std::ofstream &fout) {

		fout << _layer_type << std::endl;
		fout << _unit_count << " " << _prev_unit_count << " ";
		fout << _filter_width << " " << _filter_height << " ";
		fout << _width << " " << _height << " " << _map_num << std::endl;
		fout<<std::endl;
	}
	void read(std::ifstream &fin) {

		fin >> _layer_type;
		fin >> _unit_count >> _prev_unit_count;
		fin >> _filter_width >> _filter_height;
		fin >> _width >> _height >> _map_num;
		init();
	}
};


#endif
