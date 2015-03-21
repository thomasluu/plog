/*

Copyright 2015 Thomas Luu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

/*

File: plog.cu

Computation of the Lambert W-function by Halley's Method.
Single and double precision implementations.

Initial guesses based on:

D.A. Barry, J.-Y. Parlange, L. Li, H. Prommer, C.J. Cunningham, and 
F. Stagnitti. Analytical approximations for real values of the Lambert 
W-function. Mathematics and Computers in Simulation, 53(1):95-103, 2000.

D.A. Barry, J.-Y. Parlange, L. Li, H. Prommer, C.J. Cunningham, and 
F. Stagnitti. Erratum to analytical approximations for real values of the 
Lambert W-function. Mathematics and computers in simulation, 59(6):543-543, 
2002.

*/

#ifndef PLOG
#define PLOG

__host__ __device__ double plog(double x)
{
  if (x == 0.0) {
    return 0.0;
  }

  double w0, w1;
  if (x > 0.0) {
    w0 = log(1.2 * x / log(2.4 * x / log1p(2.4 * x)));
  } else {
    double v = 1.4142135623730950488 * sqrt(1 + 2.7182818284590452354 * x);
    double N2 = 10.242640687119285146 + 1.9797586132081854940 * v;
    double N1 = 0.29289321881345247560 * (1.4142135623730950488 + N2);
    w0 = -1.0 + v * (N2 + v) / (N2 + v + N1 * v);
  }

  while (true) {
    double e = exp(w0);
    double f = w0 * e - x;
    w1 = w0 + ((f+f) * (1.0 + w0)) / (f * (2.0 + w0) - (e+e) * (1.0 + w0) * (1.0 + w0));
    if (fabs(w0 / w1 - 1.0) < 1.4901161193847656e-8) {
      break;
    }
    w0 = w1;
  }

  return w1;
}

__host__ __device__ float plog(float x)
{
  if (x == 0.0f) {
    return 0.0f;
  }

  float w0, w1;
  if (x > 0.0f) {
    w0 = log(1.2f * x / log(2.4f * x / log1p(2.4f * x)));
  } else {
    float v = 1.4142135623730950488f * sqrt(1 + 2.7182818284590452354f * x);
    float N2 = 10.242640687119285146f + 1.9797586132081854940f * v;
    float N1 = 0.29289321881345247560f * (1.4142135623730950488f + N2);
    w0 = -1.0f + v * (N2 + v) / (N2 + v + N1 * v);
  }

  while (true) {
    float e = exp(w0);
    float f = w0 * e - x;
    w1 = w0 + ((f+f) * (1.0f + w0)) / (f * (2.0f + w0) - (e+e) * (1.0f + w0) * (1.0f + w0));
    if (fabs(w0 / w1 - 1.0f) < 0.00034526698300124390840f) {
      break;
    }
    w0 = w1;
  }

  return w1;
}

#endif
