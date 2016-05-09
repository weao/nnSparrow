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


#ifndef __NN_INPUT_LAYER__
#define __NN_INPUT_LAYER__

class nnInputLayer : public nnLayer {


public:
  nnInputLayer() : nnLayer(NULL, NULL) {

    this->_layer_type = INPUT_LAYER;
  }

  nnInputLayer(int w, int h, int ch) : nnLayer(NULL, NULL) {

    this->_unit_count = w*h;
    this->_prev_unit_count = 0;
    this->_map_num = ch;
    this->_width = w;
    this->_height = h;
    this->_layer_type = INPUT_LAYER;
    _u_a = NULL;
  }
  ~nnInputLayer() {
  }


  void init() {

    nnLayer::clear();
    _u_a = new double[_unit_count];
    //_u_delta = new double[_unit_count];
  }
  bool inputSample(double *a, int n) {
		if(n != _unit_count)
			return false;
    memcpy(_u_a, a, sizeof(double)*n);
		// for(int i=0;i<n;i++) {
		// 	_u_a[i] = a[i];
		// }
		return true;
	}

  void updateDelta() {

  }
	void backpropagation() {

  }
	void forward() {

  }
	void updateParameters(int,double,double,double) {

  }
  int getTotalUnitCount() {
    return _unit_count;
  }

  void write(std::ofstream &fout) {

    fout << _layer_type << std::endl;
    fout << _unit_count << " " << _prev_unit_count << " ";
    fout << _width << " " << _height << " " << _map_num << std::endl;
  }
  void read(std::ifstream &fin) {

    fin >> _unit_count >> _prev_unit_count;
    fin >> _width >> _height >> _map_num;
    init();
  }

};

#endif
