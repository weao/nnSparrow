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

#include "nnFLayer.hpp"

#ifndef __NN_SM_LAYER__

#define __NN_SM_LAYER__

class nnSoftmaxLayer : public nnFLayer {


public:
	nnSoftmaxLayer(nnLayer *prev=NULL) : nnFLayer(prev) {

		this->_layer_type = SOFTMAX_LAYER;
	}

	nnSoftmaxLayer(int n, nnLayer *prev, nnLayer *next = NULL) :	nnFLayer(n, SIGMOID, prev, next) {

		this->_layer_type = SOFTMAX_LAYER;
	}

	void forward() {
 		// [n, np]*[np, 1] + [n, 1]
		//_u_a = _u_W * _prev->getActivation() + _u_b;
		int n = _unit_count, np = _prev_unit_count;
		memcpy(_u_a, _u_b, n*sizeof(double));

		double *pa = _prev->getActivation();
		for(int i=0;i<n;i++) {
			double d = 0;
			for(int j=0;j<np;j++) {
				d += _u_W[i*np+j] * pa[j];
			}
			_u_a[i] += d;
		}
    double sum = 0;
    double maxv = _u_a[0];
    for(int i=1;i<n;i++) {
      if(_u_a[i] > maxv)
        maxv = _u_a[i];
    }

	  for(int i=0;i<n;i++) {
      sum += exp(_u_a[i]-maxv);
    }
    for(int i=0;i<n;i++) {
      _u_a[i] = exp(_u_a[i]-maxv) / sum;
    }
	}

};


#endif
