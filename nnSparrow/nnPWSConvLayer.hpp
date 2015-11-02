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


#ifndef __NN_PWS_CONV_LAYER__

#define __NN_PWS_CONV_LAYER__

class nnPWSConvLayer : public nnLayer {

private:
	int _filter_width;
	int _filter_height;
	int _filter_size;
	int _section_width;
	int _section_height;
	int _section_rows;
	int _section_cols;


	double* _u_conv;
	double* _u_dconv;

	double* _u_convb;
	double* _u_dconvb;


public:
	nnPWSConvLayer(nnLayer *prev=NULL) : nnLayer(prev, NULL) {
		_filter_size = 0;
		_filter_width = 0;
		_filter_height = 0;
		_section_width = 0;
		_section_height = 0;
		_section_rows = 0;
		_section_cols = 0;
		_u_convb = NULL;
		_u_conv = NULL;
		_u_dconv = NULL;
		_u_dconvb = NULL;
	}

	nnPWSConvLayer(int fw, int fh, int sw, int sh, int nm, int at, nnLayer *prev, nnLayer *next = NULL) : 	nnLayer(prev, next) {

		this->_filter_width = fw;
		this->_filter_height = fh;
		this->_filter_size = fw * fh;
		this->_map_num = nm;// * prev->getMapNum();
		this->_section_width = sw;
		this->_section_height = sh;

		this->_act_f = nnActivation::getActivation(at);
		this->_d_act_f = nnActivation::getDActivation(at);

		int w = prev ? 1 + (prev->getWidth() - fw) : 0;
		int h = prev ? 1 + (prev->getHeight() - fh) : 0;

		if(w < 0 || h < 0) {
			w = 0;
			h = 0;
		}

		this->_section_cols = ceil(double(w)/sw);
		this->_section_rows = ceil(double(h)/sh);

		this->_unit_count = w * h;
		this->_width = w;
		this->_height = h;
		this->_prev_unit_count = prev ? prev->getUnitCount() : 0;

		this->_layer_type = 0; //FWS_CONV_LAYER;

		//_u_dW = NULL;
		_u_conv = NULL;
		_u_dconv = NULL;
		_u_convb = NULL;
		_u_dconvb = NULL;

	}
	~nnPWSConvLayer() {

		if(_u_conv)
			delete [] _u_conv;
		if(_u_dconv)
			delete [] _u_dconv;
		if(_u_convb)
			delete [] _u_convb;
		if(_u_dconvb)
			delete [] _u_dconvb;
	}

	double *getDConv() {
		return _u_dconv;
	}
	double *getDConvb() {
		return _u_dconvb;
	}

	double *getConv() {
		return _u_conv;
	}
	double *getConvb() {
		return _u_convb;
	}

	void init() {

		clear();
		double rg = sqrt(6) / sqrt(_unit_count + _prev_unit_count);

		int n = _unit_count, np = _prev_unit_count, nf = _filter_size, nm = _map_num;

		_u_a = new double[n*nm];
		memset(_u_a, 0, n*nm*sizeof(double));

		_u_delta = new double[n*nm];
		memset(_u_delta, 0, n*nm*sizeof(double));


		int ns = _section_rows * _section_cols;
		//conv
		_u_conv = new double[nf*nm*ns];
		for(int i=0;i<nf*nm*ns;i++) {
			_u_conv[i] = ((double) rand() / (RAND_MAX))*2*rg - rg;
		}
		_u_dconv = new double[nf*nm*ns];
		memset(_u_dconv, 0, nf*nm*ns*sizeof(double));


		//bias
		_u_convb = new double[nm*ns];
		for(int i=0;i<nm*ns;i++) {
			_u_convb[i] = ((double) rand() / (RAND_MAX))*2*rg - rg;
		}
		_u_dconvb = new double[nm*ns];
		memset(_u_dconvb, 0, nm*ns*sizeof(double));

	}

	inline int getSection(int y, int x) {
		return y/_section_height*_section_cols + x/_section_width;
	}

	void forward() {

		//_u_a = _u_W * _prev->getActivation() + _u_convb;// [n, np]*[np, 1]
		int n = _unit_count, np = _prev_unit_count, nm = _map_num, nf = _filter_size;
		int nmp =  _prev->getMapNum();
		int pw = _prev->getWidth();
		int fw = _filter_width;
		int fh = _filter_height;

		//number of sections of a feature map
		int ns = _section_rows * _section_cols;

		double *ua = _u_a;
		double *pua = _prev->getActivation();
		double *cv = _u_conv;
		for(int mi = 0; mi < nm; mi++, ua += n, cv += nf*ns) {

			int x = 0, y = 0;
			for(int i = 0; i < n; i++, x++) {
				if(x >= _width) {
					x = 0; y++;
				}
				*(ua + i) = _u_convb[mi*ns + getSection(y,x)];
			}

			// puast - start position of feature map of pua
			for(int puast = 0; puast < np * nmp; puast += np) {

				int x = 0, y = 0;
				int step = 0;
				for(int i = 0; i < n; i++, x++, step++ ) {
					if(x >= _width) {
						step = (step / pw + 1) * pw;
						x = 0; y++;
					}
					int sec = getSection(y,x);
					//printf("%d %d %d\n", y, x, sec);
					double d = 0;
					for(int j1 = 0, j2 = 0; j2 < nf; j1 += pw, j2 += fw ) {
						for(int k = 0; k < fw; k++ ) {

						//	printf("%lf\n", (*(cv + sec*nf + j2 + k)) * (*(pua + puast + (step + j1) + k)));
							d += (*(cv + sec*nf + j2 + k)) * (*(pua + puast + (step + j1) + k));
						}
					}
					//printf("sum: %lf\n", *(ua+i));
					*(ua + i) += d;
					//printf("sum: %lf\n", *(ua+i));
				}
				//printf("\n");
			}

			_act_f(ua, n);
		}
	}
	void backpropagation(double mu) {

		//accumulate dW
		//dW = _u_delta * _prev->getActivation().transpose();
		//[n, 1] *[1, np]
		int n = _unit_count, np = _prev_unit_count, nf = _filter_size, nm = _map_num;
		//int cc = nm / _prev->getMapNum();
		int nmp = _prev->getMapNum();
		int pw = _prev->getWidth();
		int fw = _filter_width;
		int fh = _filter_height;

		//number of sections of a feature map
		int ns = _section_rows * _section_cols;

		double *dc = _u_dconv;

		double *pua = _prev->getActivation();
		double *dt = _u_delta;
		for(int mi = 0; mi < nm; mi++, dt += n, dc += nf*ns) {

			//int puast = np * (mi / cc);
			for(int puast = 0; puast < np * nmp; puast += np) {
				int step = 0, x = 0, y = 0;
				for(int i = 0; i < n; i++, x++, step++ ) {
					if(x >= _width) {
						step = (step / pw + 1) * pw;
						x = 0; y++;
					}
					int sec = getSection(y,x);
					double d = *(dt + i);
					//printf("\n%d %d %d %lf\n\n", y, x, sec, d);
					for(int j1 = 0, j2 = 0; j2 < nf; j1 += pw, j2 += fw ) {
						for(int k = 0; k < fw; k++ ) {
							*(dc + sec*nf + j2 + k) += d * (*(pua + puast + (step + j1) + k));
						//	printf("%d %lf\n", sec*nf + j2 + k, (*(pua + puast + (step + j1) + k)));
						}
					}
				}
			}
		}

		//_u_dconvb = mu*_u_dconvb + _u_delta;
		// dt = _u_delta;
		// for(int mi = 0; mi < nm; mi++, dt += n) {
		// 	double sum = 0;
		// 	for(int i=0;i<n;i++) {
		// 		sum += dt[i];
		// 	}
		// 	//_u_dconvb[mi] *= mu;
		// 	_u_dconvb[mi] += sum;
		// }

		dt = _u_delta;
		double *dcb = _u_dconvb;
		for(int mi = 0; mi < nm; mi++, dt += n, dcb += ns) {
			int x = 0, y = 0, step = 0;
			for(int i = 0; i < n; i++, x++, step++ ) {
				if(x >= _width) {
					step = (step / pw + 1) * pw;
					x = 0; y++;
				}
				int sec = getSection(y,x);
				*(dcb + sec) += *(dt + i);
			}
		}

		//t = (_u_W.transpose() * _u_delta); // [np, n] * [n, 1]
		//_prev->updateDelta();
		double *pdt = _prev->getDelta();
		if(pdt) {

			memset(pdt, 0, sizeof(double)*_prev->getTotalUnitCount());
			dt = _u_delta;
			double *cv = _u_conv;
			for(int mi = 0; mi < nm; mi++, dt += n, cv += nf*ns) {

				//int puast = np * (mi / cc);
				for(int puast = 0; puast < np * nmp; puast += np) {
					int step = 0, x = 0, y = 0;
					for(int i = 0; i < n; i++, x++, step++ ) {
						if(x >= _width) {
							step = (step / pw + 1) * pw;
							x = 0; y++;
						}
						int sec = getSection(y,x);
						double d = *(dt + i);
						for(int j1 = 0, j2 = 0; j2 < nf; j1 += pw, j2 += fw ) {
							for(int k = 0; k < fw; k++ ) {
								*(pdt + puast + (step + j1) + k) += *(cv + sec*nf + j2 + k) * d;
							}
						}
					}
				}
			}
			_prev->updateDelta();
		}
	}

	void updateDelta() {

		int n = _unit_count, nm = _map_num;
		// mat = ( W'delta )
		// f'(z), where a = f(z) is sigmoid funtion
		_d_act_f(_u_a, n*nm);
		//_u_delta = mat.cwiseProduct(df);
		for(int i = 0; i < n*nm; i++) {
			_u_delta[i] *= _u_a[i];
		}
	}

	void updateParameters(int m, double alpha, double lambda, double mu) {

		int n = _unit_count, nf = _filter_size, nm = _map_num, ns = _section_cols * _section_rows;
		double rm = 1.0 / m;

		//_u_conv = _u_conv - alpha * ( rm * _u_dconv + lambda * _u_conv );
		for(int i = 0; i < nf*nm*ns; i++) {
			_u_conv[i] -= alpha * ( rm * _u_dconv[i] + lambda * _u_conv[i] );
		}

		//_u_convb = _u_convb - alpha * (rm * _u_dconvb );
		for(int i = 0; i < nm*ns; i++) {
			_u_convb[i] -= alpha * ( rm * _u_dconvb[i] );
		}

		for(int i = 0; i < nm*nf*ns; i++) {
			_u_dconv[i] *= mu;
		}
		for(int mi = 0; mi < nm*ns; mi++) {
			_u_dconvb[mi] *= mu;
		}
	}



	int getTotalUnitCount() {
		return _unit_count * _map_num;
	}

	void clear() {
		nnLayer::clear();
		if(_u_conv) {
			delete [] _u_conv;
			_u_conv = NULL;
		}
		if(_u_convb) {
			delete [] _u_convb;
			_u_convb = NULL;
		}

		if(_u_dconv) {
			delete [] _u_dconv;
			_u_dconv = NULL;
		}
		if(_u_dconvb) {
			delete [] _u_dconvb;
			_u_dconvb = NULL;
		}
	}

	void write(std::ofstream &fout) {

		fout << _layer_type << std::endl;
		fout << _unit_count << " " << _prev_unit_count << " ";
		fout << _filter_width << " " << _filter_height << " ";
		fout << _width << " " << _height << " " << _map_num << std::endl;
		for(int i=0;i<_filter_size*_map_num;i++) {
			fout << _u_conv[i] << " ";
		}
		fout<<std::endl;
		for(int i=0;i<_map_num;i++) {
			fout << _u_convb[i] << " ";
		}
		fout<<std::endl;
	}
	void read(std::ifstream &fin) {

		fin >> _unit_count >> _prev_unit_count;
		fin >> _filter_width >> _filter_height;
		fin >> _width >> _height >> _map_num;

		_filter_size = _filter_width * _filter_height;

		init();

		for(int i=0;i<_filter_size*_map_num;i++) {
			fin >> _u_conv[i];
		}
		for(int i=0;i<_map_num;i++) {
			fin >> _u_convb[i];
		}

	}

};


#endif
