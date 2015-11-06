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

#ifndef __NN_F_LAYER__

#define __NN_F_LAYER__

class nnFLayer : public nnLayer {

protected:
	double* _u_dW;
	double* _u_db;
	double* _u_vW;
	double* _u_vb;

public:
	nnFLayer(nnLayer *prev=NULL) : nnLayer(prev, NULL) {
		_u_dW = NULL;
		_u_db = NULL;
		_u_vW = NULL;
		_u_vb = NULL;
	}

	nnFLayer(int n, int at, nnLayer *prev, nnLayer *next = NULL) :	nnLayer(prev, next) {

		this->_unit_count = n;
		this->_prev_unit_count = prev ? prev->getTotalUnitCount() : 0;
		this->_width = n;
		this->_height = 1;
		this->_map_num = 1;

		this->_act_f = nnActivation::getActivation(at);
		this->_d_act_f = nnActivation::getDActivation(at);


		_u_dW = NULL;
		_u_db = NULL;
		_u_vW = NULL;
		_u_vb = NULL;

		this->_layer_type = FULL_LAYER;
	}
	~nnFLayer() {
		if(_u_dW)
			delete [] _u_dW;
		if(_u_db)
			delete [] _u_db;
		if(_u_vW)
			delete [] _u_vW;
		if(_u_vb)
			delete [] _u_vb;
	}


	void init() {

		int np = _prev_unit_count;
		int n = _unit_count;

		clear();


		double rg = sqrt(6) / sqrt(n + np);

		_u_W = new double[n*np];
		for(int i=0;i<n*np;i++) {
			_u_W[i] = ((double) rand() / (RAND_MAX))*2*rg - rg;
		}

		_u_b = new double[n];
		for(int i=0;i<n;i++) {
			_u_b[i] = ((double) rand() / (RAND_MAX))*2*rg - rg;
		}

		_u_a = new double[n];
		memset(_u_a, 0, n*sizeof(double));

		_u_delta = new double[n];
		memset(_u_delta, 0, n*sizeof(double));

		_u_dW = new double[n*np];
		memset(_u_dW, 0, n*np*sizeof(double));

		_u_db = new double[n];
		memset(_u_db, 0, n*sizeof(double));

		_u_vW = new double[n*np];
		memset(_u_vW, 0, n*np*sizeof(double));

		_u_vb = new double[n];
		memset(_u_vb, 0, n*sizeof(double));
	}


	void forward() {
 		// [n, np]*[np, 1] + [n, 1]
		//_u_a = _u_W * _prev->getActivation() + _u_b;
		int n = _unit_count, np = _prev_unit_count;
		memcpy(_u_a, _u_b, n*sizeof(double));

		double *pua = _prev->getActivation();
		for(int i=0;i<n;i++) {
			double d = 0;
			for(int j=0;j<np;j++) {
				d += _u_W[i*np+j] * pua[j];
			}
			_u_a[i] += d;
		}
		_act_f(_u_a, n);

	}
	void backpropagation() {

		//accumulate dW, db
		//_u_dW = mu*_u_dW + _u_delta * _prev->getActivation().transpose(); [n,1] * [1,np]
		int n = _unit_count, np = _prev_unit_count;
		double *pua = _prev->getActivation();
		for(int i = 0; i < n; i++) {
			double d = _u_delta[i];
			for(int j = 0; j < np; j++) {
				//_u_dW[i*np+j] *= mu;
				_u_dW[i*np+j] += d * pua[j];
			}
		}

		//_u_db = mu*_u_db + _u_delta;
		for(int i=0;i<n;i++) {
			//_u_db[i] *= mu;
		 	_u_db[i] += _u_delta[i];
		}

		//t = (_u_W.transpose() * _u_delta); // [np, n] * [n, 1]
		double *pdt = _prev->getDelta();
		if(pdt) {

			memset(pdt, 0, np*sizeof(double));
			for(int j = 0; j < n; j++) {
				double d = _u_delta[j];
				for(int i = 0; i < np; i++) {
					pdt[i] += _u_W[j*np+i] * d;
				}
			}

			_prev->updateDelta();
		}


	}
	void updateParameters(int m, double alpha, double lambda, double mu) {


		int n = _unit_count, np = _prev_unit_count;
		double rm = 1.0 / m;

		//_u_W = _u_W - alpha * ( rm * _u_dW + lambda * _u_W );
		for(int i=0;i<n*np;i++) {
			_u_vW[i] = _u_vW[i] * mu + alpha * (rm * _u_dW[i] + lambda * _u_W[i]);
			_u_W[i] -= _u_vW[i];
			//_u_W[i] -= alpha * (rm * _u_dW[i] + lambda * _u_W[i]);
		}

		//_u_b = _u_b - alpha * ( rm * _u_db );
		for(int i=0;i<n;i++) {
			//_u_b[i] -= alpha * (rm * _u_db[i]);
			_u_vb[i] = _u_vb[i] * mu + alpha * (rm * _u_db[i]);
			_u_b[i] -= _u_vb[i];
			//printf("%lf ", _u_b[i] );
		}

		for(int i = 0; i < n; i++) {
			for(int j = 0; j < np; j++) {
				_u_dW[i*np+j] = 0;
			}
		}
		for(int i=0;i<n;i++) {
			_u_db[i] = 0;
		}

	}
	void updateDelta() {

		// mat = ( W'delta )
		// f'(z), where a = f(z) is sigmoid funtion

		_d_act_f(_u_a, _unit_count);

		for(int i=0;i<_unit_count;i++) {
			_u_delta[i] *= _u_a[i];
		}
	}

	bool calculateDelta(double *result, int n) {
		if(n != _unit_count)
			return false;

		for(int i=0;i<_unit_count;i++)
			_u_delta[i] = -(result[i] - _u_a[i]);


		return true;
	}

	int getTotalUnitCount() {
		return _unit_count;
	}

	void clear() {
		nnLayer::clear();

		if(_u_dW) {
			delete [] _u_dW;
			_u_dW = NULL;
		}
		if(_u_db) {
			delete [] _u_db;
			_u_db = NULL;
		}
		if(_u_vW) {
			delete [] _u_vW;
			_u_vW = NULL;
		}
		if(_u_vb) {
			delete [] _u_vb;
			_u_vb = NULL;
		}

	}
	void write(std::ofstream &fout) {

		fout << _layer_type << std::endl;
		fout << _unit_count << " " << _prev_unit_count << " ";
		fout << _width << " " << _height << " " << _map_num << std::endl;

		for(int i=0;i<_unit_count;i++) {
			for(int j=0;j<_prev_unit_count;j++)
				fout << _u_W[i*_prev_unit_count+j] << " ";
			fout<<std::endl;
		}
		for(int i=0;i<_unit_count;i++) {
			fout << _u_b[i] << " ";
		}
		fout<<std::endl;
	}
	void read(std::ifstream &fin) {

		fin >> _unit_count >> _prev_unit_count;
		fin >> _width >> _height >> _map_num;

		init();

		for(int i=0;i<_unit_count;i++) {
			for(int j=0;j<_prev_unit_count;j++)
				fin >> _u_W[i*_prev_unit_count+j];
		}
		for(int i=0;i<_unit_count;i++) {
			fin >> _u_b[i];
		}
	}

};


#endif
