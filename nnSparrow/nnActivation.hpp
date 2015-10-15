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
#include <cstdlib>
#include <cstdio>
#include <cfloat>

#ifndef __NN_ACTIVATION__
#define __NN_ACTIVATION__

typedef void (*activation)(double*, int);

enum ACTIVATION_TYPE {
  SIGMOID = 0,
  TANH = 1,
  RECTIFIER = 2
};

class nnActivation {

public:
  static activation getActivation(int type) {

    static activation f[] = {nnActivation::sigmoid, nnActivation::tanh, nnActivation::rectifier};
    return f[type];
  }
  static activation getDActivation(int type) {

    static activation df[] = {nnActivation::dsigmoid, nnActivation::dtanh, nnActivation::drectifier};
    return df[type];
  }


  static void sigmoid(double *a, int n) {

    for(int i=0;i<n;i++) {
      a[i] = 1.0 / ( exp(-a[i]) + 1.0 );
    }
  }
  static void dsigmoid(double *a, int n) {

    for(int i=0;i<n;i++) {
      a[i] = a[i] * ( 1.0 - a[i] );
    }
  }

  static void tanh(double *a, int n) {

    for(int i=0;i<n;i++) {
      double x1 = exp(a[i]);
      double x2 = exp(-a[i]);
       a[i] = (x1 - x2) / (x1 + x2);
    }
  }

  static void dtanh(double *a, int n) {

    for(int i=0;i<n;i++) {
      a[i] = 1 - a[i]*a[i];
    }
  }

  static void rectifier(double *a, int n) {

    for(int i=0;i<n;i++) {
      if(a[i] < 0)
          a[i] = 0;
    }
  }

  static void drectifier(double *a, int n) {
    for(int i=0;i<n;i++) {
      a[i] = (a[i] > 0) ? 1 : 0;
    }
  }

};

#endif
