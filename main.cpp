#include <iostream>
#include <chrono>
#include <cmath>
// #include "gnuplot-iostream.h"
#include "Vec.h"
#include "Mat.h"
#include "tests.h"
#include "tools.h"

void tests() {
  auto start = std::chrono::high_resolution_clock::now();
  tests_Vec();
  tests_Mat();
  tests_MP();
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
}

Mat f(Mat x) {
  double c = 0.1;
  return -x.unit()*c/(std::pow(x.norm(),2));
}

int main() {
  tests();

};
