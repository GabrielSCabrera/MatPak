#include <iostream>
#include <cassert>
#include "Vec.h"
#include "Mat.h"
#include "tools.h"

void tests_Vec() {

  // Creating two new Vec objects
  Vec u(5);
  Vec v(5);
  Vec z(10);

  // Checking their lengths
  assert(u.length() == 5);
  assert(v.length() == 5);
  assert(z.length() == 10);

  // Setting "u" to [1, 2, 0, 0, 0]
  u.set(1, 0);
  u.set(2, 1);

  // Setting "v" to [0, 3, 0, 2, 0]
  v.set(2, 3);
  v.set(3, 1);

  // Setting "z" to [0, -4, 7, 0, 0, 3, 0, -1, 2, 0]
  z.set(-4, 1);
  z.set(7, 2);
  z.set(3, 5);
  z.set(-1, 7);
  z.set(2, 8);

  // Checking that the indices of "u" match the expected values
  double u_exp[5] = {1, 2, 0, 0, 0};
  for (int i = 0; i < 5; i++) {
    double diff = u.get(i) - u_exp[i];
    assert(diff == 0);
  }

  // Checking that the indices of "v" match the expected values
  double v_exp[5] = {0, 3, 0, 2, 0};
  for (int i = 0; i < 5; i++) {
    double diff = v.get(i) - v_exp[i];
    assert(diff == 0);
  }

  // Checking that the indices of "z" match the expected values
  double z_exp[10] = {0, -4, 7, 0, 0, 3, 0, -1, 2};
  for (int i = 0; i < 10; i++) {
    double diff = z.get(i) - z_exp[i];
    assert(diff == 0);
  }

  // Checking that the indices of "w" match the expected values
  Vec w = u+v;
  double w_exp[5] = {1, 5, 0, 2, 0};
  for (int i = 0; i < 5; i++) {
    double diff = w.get(i) - w_exp[i];
    assert(diff == 0);
  }

  // Checking that the indices of "x" match the expected values
  Vec x = u-v;
  double x_exp[5] = {1, -1, 0, -2, 0};
  for (int i = 0; i < 5; i++) {
    double diff = x.get(i) - x_exp[i];
    assert(diff == 0);
  }

  // Checking that "y" is equal to the expected value
  double y = x.inner(w);
  assert(y == -8);

  // Testing multiplication
  Vec g(3);
  g.set(1, 0);
  g.set(4, 1);
  g.set(-3, 2);
  g = g*(-2);
  double g_exp[3] = {-2, -8, 6};
  for (int i = 0; i < 3; i++) {
    double diff = g.get(i) - g_exp[i];
    assert(diff == 0);
  }

  // Testing the cross product
  Vec a(3);
  a.set(3, 0);
  a.set(-3, 1);
  a.set(1, 2);
  Vec b(3);
  b.set(4, 0);
  b.set(9, 1);
  b.set(2, 2);
  Vec c = a.cross(b);
  assert(c.get(0) == -15);
  assert(c.get(1) == -2);
  assert(c.get(2) == 39);

  // Confirming the string output of "u" is correct
  std::string u_exp_str = "[1.00E+00, 2.00E+00, 0.00E+00, 0.00E+00, 0.00E+00]";
  assert(u.string().compare(u_exp_str) == 0);

  // // Printing out "u"
  // std::cout << "The pairs must match, if not there is an error" << std::endl << std::endl;
  // std::cout << "[1.00E+00, 2.00E+00, 0.00E+00, 0.00E+00, 0.00E+00] = u" << std::endl;
  // u.print();

  // Confirming the string output of "v" is correct
  std::string v_exp_str = "[0.00E+00, 3.00E+00, 0.00E+00, 2.00E+00, 0.00E+00]";
  assert(v.string().compare(v_exp_str) == 0);

  // // Printing out "v"
  // std::cout << std::endl << "[0.00E+00, 3.00E+00, 0.00E+00, 2.00E+00, 0.00E+00] = v" << std::endl;
  // v.print();

  // Confirming the string output of "w" is correct
  std::string w_exp_str = "[1.00E+00, 5.00E+00, 0.00E+00, 2.00E+00, 0.00E+00]";
  assert(w.string().compare(w_exp_str) == 0);

  // // Printing out "w"
  // std::cout << std::endl << "[1.00E+00, 5.00E+00, 0.00E+00, 2.00E+00, 0.00E+00] = w" << std::endl;
  // w.print();

  // Confirming the string output of "x" is correct
  std::string x_exp_str = "[1.00E+00, -1.00E+00, 0.00E+00, -2.00E+00, 0.00E+00]";
  assert(x.string().compare(x_exp_str) == 0);

  // // Printing out "x"
  // std::cout << std::endl << "[1.00E+00, -1.00E+00, 0.00E+00, -2.00E+00, 0.00E+00] = x" << std::endl;
  // x.print();

  // Testing inplace addition of "u" and "v", comparing to "w" elementwise
  u += v;
  for (int i = 0; i < 5; i++) {
    double diff = u.get(i) - w_exp[i];
    assert(diff == 0);
  }

  // Testing inplace subtraction of "u" and "v", comparing to "w" elementwise
  u -= v;
  for (int i = 0; i < 5; i++) {
    double diff = u.get(i) - u_exp[i];
    assert(diff == 0);
  }

  // Testing inplace multiplication of "h" with a double
  Vec h(4);
  h.set(-3, 0);
  h.set(100, 1);
  h.set(-3.14, 2);
  h.set(345.02, 3);

  double h_exp[4] = {6, -200, 6.28, -690.04};
  for (int i = 0; i < 4; i++) {
    double diff = -2*h.get(i) - h_exp[i];
    assert(diff == 0);
  }

  // Checking that the "is equal" operator "==" works properly
  assert((u == u) == true);
  assert((u == v) == false);

  // Checking that the "is not equal" operator "!=" works properly
  assert((u != u) == false);
  assert((u != v) == true);

  // Testing the negative operator
  Vec d(3);
  d.set(1.235, 0);
  d.set(-235.0001, 1);
  d.set(23.124, 2);
  Vec e(3);
  e.set(-1.235, 0);
  e.set(235.0001, 1);
  e.set(-23.124, 2);
  assert(-e == d);


  // Testing the "norm()" function
  Vec f(5);
  f.set(-1, 0);
  f.set(1, 1);
  f.set(-2, 2);
  f.set(3, 3);
  f.set(1, 4);
  assert(f.norm() == 4);

  // Testing the unit vector method "unit()"
  double f_exp[5] = {-0.25, 0.25, -0.5, 0.75, 0.25};
  f = f.unit();
  for (int i = 0; i < 5; i++) {
    assert(f_exp[i] == f.get(i));
  }

};

void tests_Mat() {

  // Testing 1-D matrix generation
  Mat a1(2);
  Vec b1(1);
  b1.set(2, 0);
  assert(a1.length() == 2);
  assert(a1.dimension() == 1);
  assert(b1 == a1.shape());

  // Testing 2-D matrix generation
  Mat a2(2,3);
  Vec b2(2);
  b2.set(2, 0);
  b2.set(3, 1);
  assert(a2.length() == 6);
  assert(a2.dimension() == 2);
  assert(b2 == a2.shape());

  // Testing 3-D matrix generation
  Mat a3(1,5,2);
  Vec b3(3);
  b3.set(1, 0);
  b3.set(5, 1);
  b3.set(2, 2);
  assert(a3.length() == 10);
  assert(a3.dimension() == 3);
  assert(b3 == a3.shape());

  // Testing 4-D matrix generation
  Mat a4(2,5,4,3);
  Vec b4(4);
  b4.set(2, 0);
  b4.set(5, 1);
  b4.set(4, 2);
  b4.set(3, 3);
  assert(a4.length() == 120);
  assert(a4.dimension() == 4);
  assert(b4 == a4.shape());

  // Testing 5-D matrix generation
  Mat a5(1,4,2,3,5);
  Vec b5(5);
  b5.set(1, 0);
  b5.set(4, 1);
  b5.set(2, 2);
  b5.set(3, 3);
  b5.set(5, 4);
  assert(a5.length() == 120);
  assert(a5.dimension() == 5);
  assert(b5 == a5.shape());

  // Testing 6-D matrix generation
  Mat a6(1,3,4,2,3,3);
  Vec b6(6);
  b6.set(1, 0);
  b6.set(3, 1);
  b6.set(4, 2);
  b6.set(2, 3);
  b6.set(3, 4);
  b6.set(3, 5);
  assert(a6.length() == 216);
  assert(a6.dimension() == 6);
  assert(b6 == a6.shape());

  // Testing 1-D matrix addition

  Mat c11(2);
  c11 = "[1, 4]";

  Mat c21(2);
  c21 = "[8,-5]";

  Mat c31(2);
  c31 = "[9,-1]";
  assert(c11 + c21 == c31);

  // Testing 2-D matrix addition

  Mat c12(3, 2);
  c12 = "[[8,-6],[-4,2],[-6,4]]";

  Mat c22(3, 2);
  c22 = "[[2,8],[-3,-5],[-1,0]]";

  Mat c32(3, 2);
  c32 = "[[10,2],[-7,-3],[-7,4]]";

  assert(c12 + c22 == c32);
  // Testing 3-D matrix addition

  Mat c13(3, 3, 3);
  c13 = "[[[1,-9,4],[-4,0,-9],[-6,-4,-8]],[[-7,-4,4],[7,-2,8],[0,-1,6]],[[-8,3,-2],[4,7,3],[-9,-7,1]]]";

  Mat c23(3, 3, 3);
  c23 = "[[[-8,-1,-1],[-1,4,5],[-4,4,-1]],[[-9,7,2],[-6,4,-5],[-7,8,-2]],[[-7,-4,-9],[-9,1,-8],[-1,-4,8]]]";

  Mat c33(3, 3, 3);
  c33 = "[[[-7,-10,3],[-5,4,-4],[-10,0,-9]],[[-16,3,6],[1,2,3],[-7,7,4]],[[-15,-1,-11],[-5,8,-5],[-10,-11,9]]]";

  assert(c13 + c23 == c33);

  // Testing 4-D matrix addition

  Mat c14(3, 3, 3, 3);
  c14 = "[[[[-4,4,7],[-2,7,-7],[2,7,-5]],[[8,-4,2],[-6,-8,-6],[-7,4,-2]],[[2,-7,3],[-6,7,3],[-3,-8,5]]],[[[-4,-7,4],[-8,8,-4],[4,-7,-5]],[[-5,8,6],[0,-3,8],[-9,0,4]],[[2,-8,-7],[-6,4,6],[3,3,-8]]],[[[-1,-2,-4],[-7,2,-7],[-9,-1,-1]],[[5,0,-2],[-1,7,2],[-9,2,-3]],[[1,-7,-3],[6,5,-7],[8,8,-6]]]]";

  Mat c24(3, 3, 3, 3);
  c24 = "[[[[-4,-3,1],[8,2,-5],[5,2,-2]],[[2,8,-1],[-1,-1,6],[-1,8,7]],[[-5,-7,8],[-4,-3,-5],[1,4,7]]],[[[-9,4,-5],[-1,1,8],[1,-5,-3]],[[-3,-7,-9],[-3,8,4],[1,3,-7]],[[-9,-7,-6],[-9,-2,-5],[7,1,-8]]],[[[-7,-8,8],[-5,6,0],[8,-2,4]],[[7,-7,-5],[-5,-8,7],[-1,-7,2]],[[0,5,1],[-1,0,-2],[-1,6,-3]]]]";

  Mat c34(3, 3, 3, 3);
  c34 = "[[[[-8,1,8],[6,9,-12],[7,9,-7]],[[10,4,1],[-7,-9,0],[-8,12,5]],[[-3,-14,11],[-10,4,-2],[-2,-4,12]]],[[[-13,-3,-1],[-9,9,4],[5,-12,-8]],[[-8,1,-3],[-3,5,12],[-8,3,-3]],[[-7,-15,-13],[-15,2,1],[10,4,-16]]],[[[-8,-10,4],[-12,8,-7],[-1,-3,3]],[[12,-7,-7],[-6,-1,9],[-10,-5,-1]],[[1,-2,-2],[5,5,-9],[7,14,-9]]]]";

  assert(c14 + c24 == c34);

  // Testing 5-D matrix addition

  Mat c15(3, 3, 3, 3, 3);
  c15 = "[[[[[-4,5,4],[-2,-6,3],[8,1,5]],[[-5,3,8],[-9,0,-6],[-6,-8,-6]],[[0,-9,-4],[-5,-7,5],[2,6,6]]],[[[5,7,-6],[2,0,7],[4,-7,-8]],[[3,-6,7],[-6,2,-5],[0,5,-4]],[[8,-9,2],[-5,-7,-4],[-7,-9,-6]]],[[[4,4,-2],[3,6,-2],[4,4,-9]],[[8,-4,0],[-4,3,-5],[-1,-4,2]],[[7,7,-6],[-4,-2,1],[4,-7,7]]]],[[[[0,-9,-8],[-9,-7,4],[-9,-3,0]],[[-1,-4,-9],[3,-4,-2],[5,-8,3]],[[-9,-5,4],[-8,-2,-6],[2,-4,3]]],[[[-3,-1,-5],[-1,-8,-2],[-8,-2,-2]],[[-9,-6,1],[-2,-4,8],[6,-7,-9]],[[-7,-4,7],[-9,-4,4],[-1,-1,4]]],[[[7,-3,1],[4,7,7],[4,1,7]],[[5,7,5],[-1,3,1],[-8,3,-2]],[[3,5,7],[-2,1,-3],[3,0,-3]]]],[[[[4,0,-2],[-4,-9,1],[-9,-2,2]],[[3,-9,4],[1,-1,-3],[-9,-5,-8]],[[-9,0,1],[-3,-1,7],[-5,4,-9]]],[[[-1,4,6],[3,-8,-4],[7,7,-7]],[[-7,2,4],[8,-6,-7],[-1,4,2]],[[-5,1,-3],[1,6,-9],[-3,8,1]]],[[[-1,5,-7],[-5,-6,-5],[-4,3,8]],[[-9,7,-5],[-6,2,-4],[3,-4,1]],[[-2,5,-8],[2,-2,-5],[8,6,-8]]]]]";

  Mat c25(3, 3, 3, 3, 3);
  c25 = "[[[[[0,0,-7],[-8,-5,-4],[-9,2,-4]],[[5,-1,3],[4,-8,1],[6,1,-1]],[[2,7,-6],[7,7,-9],[-5,-3,4]]],[[[4,8,6],[6,0,0],[4,-8,-9]],[[0,-9,5],[-4,-2,-7],[-1,-9,-4]],[[5,-1,5],[7,-7,-2],[6,5,-2]]],[[[-6,-3,2],[-7,4,3],[1,-7,-6]],[[7,3,-1],[-2,7,-1],[0,4,-1]],[[-1,3,-2],[-6,-3,-1],[4,-7,6]]]],[[[[2,1,-3],[0,-5,-6],[-4,-4,-6]],[[-6,-9,-7],[-7,-2,-1],[0,-4,4]],[[2,-5,-3],[-6,2,8],[8,2,3]]],[[[-6,1,-4],[4,-3,-4],[-9,-2,-4]],[[-2,5,2],[5,-5,1],[8,-9,1]],[[-9,2,6],[-1,5,7],[1,-2,1]]],[[[-1,-9,4],[-7,-6,-1],[0,-8,-8]],[[4,3,-7],[3,5,-4],[-5,-4,-7]],[[-9,-2,3],[-6,0,-4],[8,3,-6]]]],[[[[2,3,6],[4,7,1],[-9,2,1]],[[4,-3,-1],[-9,2,0],[-1,3,-2]],[[4,6,8],[-5,4,-7],[1,7,-9]]],[[[3,-1,4],[3,0,3],[-1,6,2]],[[-1,-3,-5],[-7,-5,1],[-4,4,1]],[[2,-1,-1],[-5,7,3],[3,8,8]]],[[[6,7,7],[-8,-2,2],[-9,-4,-1]],[[2,8,3],[3,1,5],[-3,7,3]],[[8,-8,-8],[8,-7,0],[-1,-9,4]]]]]";

  Mat c35(3, 3, 3, 3, 3);
  c35 = "[[[[[-4,5,-3],[-10,-11,-1],[-1,3,1]],[[0,2,11],[-5,-8,-5],[0,-7,-7]],[[2,-2,-10],[2,0,-4],[-3,3,10]]],[[[9,15,0],[8,0,7],[8,-15,-17]],[[3,-15,12],[-10,0,-12],[-1,-4,-8]],[[13,-10,7],[2,-14,-6],[-1,-4,-8]]],[[[-2,1,0],[-4,10,1],[5,-3,-15]],[[15,-1,-1],[-6,10,-6],[-1,0,1]],[[6,10,-8],[-10,-5,0],[8,-14,13]]]],[[[[2,-8,-11],[-9,-12,-2],[-13,-7,-6]],[[-7,-13,-16],[-4,-6,-3],[5,-12,7]],[[-7,-10,1],[-14,0,2],[10,-2,6]]],[[[-9,0,-9],[3,-11,-6],[-17,-4,-6]],[[-11,-1,3],[3,-9,9],[14,-16,-8]],[[-16,-2,13],[-10,1,11],[0,-3,5]]],[[[6,-12,5],[-3,1,6],[4,-7,-1]],[[9,10,-2],[2,8,-3],[-13,-1,-9]],[[-6,3,10],[-8,1,-7],[11,3,-9]]]],[[[[6,3,4],[0,-2,2],[-18,0,3]],[[7,-12,3],[-8,1,-3],[-10,-2,-10]],[[-5,6,9],[-8,3,0],[-4,11,-18]]],[[[2,3,10],[6,-8,-1],[6,13,-5]],[[-8,-1,-1],[1,-11,-6],[-5,8,3]],[[-3,0,-4],[-4,13,-6],[0,16,9]]],[[[5,12,0],[-13,-8,-3],[-13,-1,7]],[[-7,15,-2],[-3,3,1],[0,3,4]],[[6,-3,-16],[10,-9,-5],[7,-3,-4]]]]]";

  assert(c15 + c25 == c35);

  // Testing 1-D matrix subtraction

  Mat d11(3);
  d11 = "[3,6,7]";

  Mat d21(3);
  d21 = "[1,-9,-2]";

  Mat d31(3);
  d31 = "[2,15,9]";

  assert(d11 - d21 == d31);

  // Testing 2-D matrix subtraction

  Mat d12(2, 3);
  d12 = "[[-9,-7,2],[-2,-2,-1]]";

  Mat d22(2, 3);
  d22 = "[[-2,-9,-6],[7,8,-4]]";

  Mat d32(2, 3);
  d32 = "[[-7,2,8],[-9,-10,3]]";

  assert(d12 - d22 == d32);

  // Testing 3-D matrix subtraction

  Mat d13(3, 3, 3);
  d13 = "[[[-1,2,5],[3,2,-8],[4,6,-7]],[[-4,1,6],[1,-3,8],[-6,1,-7]],[[-8,-9,0],[-9,3,-5],[0,3,-7]]]";

  Mat d23(3, 3, 3);
  d23 = "[[[6,-3,7],[6,-5,-4],[0,-7,-7]],[[3,-7,0],[5,8,3],[-8,0,1]],[[2,1,5],[7,-6,-8],[-2,-1,-6]]]";

  Mat d33(3, 3, 3);
  d33 = "[[[-7,5,-2],[-3,7,-4],[4,13,0]],[[-7,8,6],[-4,-11,5],[2,1,-8]],[[-10,-10,-5],[-16,9,3],[2,4,-1]]]";

  assert(d13 - d23 == d33);

  // Testing 4-D matrix subtraction

  Mat d14(2, 3, 3, 3);
  d14 = "[[[[0,0,-5],[-8,-9,-6],[8,-2,5]],[[-5,-1,3],[-6,1,-9],[-6,2,-6]],[[3,6,-8],[-9,3,0],[6,8,7]]],[[[3,2,4],[-6,-7,7],[2,5,-9]],[[-2,-4,5],[-1,7,-4],[0,1,8]],[[4,-3,-1],[-4,-5,-8],[6,-6,7]]]]";

  Mat d24(2, 3, 3, 3);
  d24 = "[[[[-4,8,-3],[1,-6,1],[5,-4,4]],[[-3,-6,3],[1,-6,3],[-3,-8,1]],[[8,-7,-7],[-7,7,-5],[5,-8,-5]]],[[[-1,-6,-1],[4,-7,-6],[2,6,-6]],[[-3,8,7],[-8,-7,8],[7,6,-9]],[[-8,6,2],[3,2,1],[-7,0,7]]]]";

  Mat d34(2, 3, 3, 3);
  d34 = "[[[[4,-8,-2],[-9,-3,-7],[3,2,1]],[[-2,5,0],[-7,7,-12],[-3,10,-7]],[[-5,13,-1],[-2,-4,5],[1,16,12]]],[[[4,8,5],[-10,0,13],[0,-1,-3]],[[1,-12,-2],[7,14,-12],[-7,-5,17]],[[12,-9,-3],[-7,-7,-9],[13,-6,0]]]]";

  assert(d14 - d24 == d34);

  // Testing 5-D matrix subtraction

  Mat d15(2, 3, 3, 3, 3);
  d15 = "[[[[[-5,-7,5],[3,-8,-6],[-6,-4,6]],[[-6,4,8],[4,1,4],[3,-1,-8]],[[-1,7,7],[-1,5,6],[2,-4,0]]],[[[4,-8,-9],[2,-8,2],[-9,4,-8]],[[1,8,5],[-6,-6,-7],[0,6,7]],[[-2,1,-8],[4,-2,-1],[8,-6,-7]]],[[[3,5,3],[-5,1,-1],[-7,4,8]],[[1,2,-8],[-1,-1,5],[4,3,-2]],[[-1,-8,6],[8,-7,7],[-7,5,-2]]]],[[[[-6,-9,0],[4,8,-2],[0,-6,-9]],[[6,-1,3],[5,-8,-6],[7,-9,5]],[[-4,-3,-2],[-7,3,-7],[-7,-4,7]]],[[[4,1,5],[1,-2,-3],[-3,8,2]],[[-9,5,6],[7,-6,0],[-6,-9,-4]],[[5,-1,-7],[2,-1,2],[1,8,-2]]],[[[0,0,2],[6,-6,3],[0,-1,4]],[[-2,-9,-5],[-1,7,5],[-7,1,8]],[[1,1,3],[-8,8,-1],[-6,4,-8]]]]]";

  Mat d25(2, 3, 3, 3, 3);
  d25 = "[[[[[6,5,6],[-7,1,1],[-7,-2,-5]],[[-7,-4,5],[4,-5,7],[6,5,-3]],[[-2,-1,1],[-7,-2,5],[-1,-3,-5]]],[[[8,4,6],[-6,-5,5],[1,-7,6]],[[8,-1,6],[2,-8,6],[-6,-4,8]],[[-8,-9,-6],[4,-4,4],[5,-5,-6]]],[[[-7,-2,-4],[-6,-9,2],[3,3,-7]],[[-1,-9,8],[3,-1,1],[0,4,-9]],[[-6,-9,-7],[-6,-1,0],[-8,-4,8]]]],[[[[-2,2,7],[4,-8,-9],[-4,-8,2]],[[-5,6,5],[-1,4,-1],[3,6,-5]],[[-9,-3,0],[5,-5,-9],[2,-7,-1]]],[[[-3,3,4],[1,0,-9],[1,1,3]],[[-9,-6,-9],[-8,0,6],[2,7,-1]],[[3,4,-8],[2,4,-3],[8,6,3]]],[[[-2,0,0],[-9,-9,-5],[0,-2,-9]],[[-2,1,2],[-4,5,-5],[7,-1,4]],[[-2,-4,7],[-5,-5,-2],[-2,0,3]]]]]";

  Mat d35(2, 3, 3, 3, 3);
  d35 = "[[[[[-11,-12,-1],[10,-9,-7],[1,-2,11]],[[1,8,3],[0,6,-3],[-3,-6,-5]],[[1,8,6],[6,7,1],[3,-1,5]]],[[[-4,-12,-15],[8,-3,-3],[-10,11,-14]],[[-7,9,-1],[-8,2,-13],[6,10,-1]],[[6,10,-2],[0,2,-5],[3,-1,-1]]],[[[10,7,7],[1,10,-3],[-10,1,15]],[[2,11,-16],[-4,0,4],[4,-1,7]],[[5,1,13],[14,-6,7],[1,9,-10]]]],[[[[-4,-11,-7],[0,16,7],[4,2,-11]],[[11,-7,-2],[6,-12,-5],[4,-15,10]],[[5,0,-2],[-12,8,2],[-9,3,8]]],[[[7,-2,1],[0,-2,6],[-4,7,-1]],[[0,11,15],[15,-6,-6],[-8,-16,-3]],[[2,-5,1],[0,-5,5],[-7,2,-5]]],[[[2,0,2],[15,3,8],[0,1,13]],[[0,-10,-7],[3,2,10],[-14,2,4]],[[3,5,-4],[-3,13,1],[-4,4,-11]]]]]";

  assert(d15 - d25 == d35);

  // Testing 1-D matrix-scalar multiplication

  Mat e11(2);
  e11 = "[-7,6]";

  Mat e31(2);
  e31 = "[28,-24]";

  assert(e11*(-4) == e31);

  // Testing 2-D matrix-scalar multiplication

  Mat e12(2, 3);
  e12 = "[[4,5,8],[2,8,-8]]";

  Mat e32(2, 3);
  e32 = "[[12,15,24],[6,24,-24]]";

  assert(e12*(3) == e32);

  // Testing 3-D matrix-scalar multiplication

  Mat e13(3, 2, 2);
  e13 = "[[[-9,3],[4,8]],[[2,2],[7,0]],[[4,1],[4,8]]]";

  Mat e33(3, 2, 2);
  e33 = "[[[18,-6],[-8,-16]],[[-4,-4],[-14,0]],[[-8,-2],[-8,-16]]]";

  assert(e13*(-2) == e33);

  // Testing 4-D matrix-scalar multiplication

  Mat e14(2, 2, 3, 2);
  e14 = "[[[[-1,-6],[1,4],[7,5]],[[6,6],[-4,-9],[0,1]]],[[[-7,-6],[-2,-5],[-6,6]],[[5,8],[0,-7],[6,3]]]]";

  Mat e34(2, 2, 3, 2);
  e34 = "[[[[-4,-24],[4,16],[28,20]],[[24,24],[-16,-36],[0,4]]],[[[-28,-24],[-8,-20],[-24,24]],[[20,32],[0,-28],[24,12]]]]";

  assert(e14*(4) == e34);

  // Testing 5-D matrix-scalar multiplication

  Mat e15(2, 3, 3, 2, 2);
  e15 = "[[[[[-3,6],[-2,-2]],[[-8,0],[-5,4]],[[-4,0],[0,6]]],[[[1,-8],[-1,0]],[[-9,-9],[4,6]],[[-9,-3],[-7,6]]],[[[-8,1],[5,-5]],[[-8,-3],[-4,-4]],[[-8,-7],[-3,2]]]],[[[[-2,8],[4,8]],[[-6,5],[3,-8]],[[-5,0],[2,6]]],[[[-9,0],[-8,6]],[[4,4],[-8,-2]],[[4,6],[6,-5]]],[[[2,5],[2,6]],[[-5,-2],[4,-6]],[[-9,-3],[8,4]]]]]";

  Mat e35(2, 3, 3, 2, 2);
  e35 = "[[[[[-12,24],[-8,-8]],[[-32,0],[-20,16]],[[-16,0],[0,24]]],[[[4,-32],[-4,0]],[[-36,-36],[16,24]],[[-36,-12],[-28,24]]],[[[-32,4],[20,-20]],[[-32,-12],[-16,-16]],[[-32,-28],[-12,8]]]],[[[[-8,32],[16,32]],[[-24,20],[12,-32]],[[-20,0],[8,24]]],[[[-36,0],[-32,24]],[[16,16],[-32,-8]],[[16,24],[24,-20]]],[[[8,20],[8,24]],[[-20,-8],[16,-24]],[[-36,-12],[32,16]]]]]";

  assert(e15*(4) == e35);

  // Testing 1-D matrix-scalar division

  Mat f11(2);
  f11 = "[-1,2]";

  Mat f31(2);
  f31 = "[-0.5,1.]";

  assert(f11/(2) == f31);

  // Testing 2-D matrix-scalar division

  Mat f12(2, 3);
  f12 = "[[2,8,1],[-6,8,0]]";

  Mat f32(2, 3);
  f32 = "[[1.,4.,0.5],[-3.,4.,0.]]";

  assert(f12/(2) == f32);

  // Testing 3-D matrix-scalar division

  Mat f13(3, 3, 2);
  f13 = "[[[-5,-2],[-4,7],[-8,-6]],[[3,5],[2,-6],[8,-7]],[[5,8],[3,-1],[7,6]]]";

  Mat f33(3, 3, 2);
  f33 = "[[[-5.,-2.],[-4.,7.],[-8.,-6.]],[[3.,5.],[2.,-6.],[8.,-7.]],[[5.,8.],[3.,-1.],[7.,6.]]]";

  assert(f13/(1) == f33);

  // Testing 4-D matrix-scalar division

  Mat f14(2, 2, 3, 2);
  f14 = "[[[[0,8],[6,-9],[2,7]],[[-6,-3],[2,3],[-8,5]]],[[[5,4],[8,-2],[2,3]],[[-6,-8],[-3,7],[7,-4]]]]";

  Mat f34(2, 2, 3, 2);
  f34 = "[[[[0.,4.],[3.,-4.5],[1.,3.5]],[[-3.,-1.5],[1.,1.5],[-4.,2.5]]],[[[2.5,2.],[4.,-1.],[1.,1.5]],[[-3.,-4.],[-1.5,3.5],[3.5,-2.]]]]";

  assert(f14/(2) == f34);

  // Testing 5-D matrix-scalar division

  Mat f15(2, 3, 2, 2, 3);
  f15 = "[[[[[5,-9,-9],[-3,-2,-5]],[[8,1,3],[0,-6,5]]],[[[5,7,-1],[6,0,-8]],[[-4,6,3],[-2,6,-9]]],[[[1,-9,-8],[-2,7,-9]],[[7,1,4],[-4,-5,1]]]],[[[[-6,0,6],[1,2,-9]],[[-3,-5,1],[-6,-2,3]]],[[[-5,5,-8],[-9,-7,5]],[[3,-1,-2],[6,-6,-5]]],[[[-3,-1,-2],[7,-2,-1]],[[-1,2,-9],[-6,1,2]]]]]";

  Mat f35(2, 3, 2, 2, 3);
  f35 = "[[[[[2.5,-4.5,-4.5],[-1.5,-1.,-2.5]],[[4.,0.5,1.5],[0.,-3.,2.5]]],[[[2.5,3.5,-0.5],[3.,0.,-4.]],[[-2.,3.,1.5],[-1.,3.,-4.5]]],[[[0.5,-4.5,-4.],[-1.,3.5,-4.5]],[[3.5,0.5,2.],[-2.,-2.5,0.5]]]],[[[[-3.,0.,3.],[0.5,1.,-4.5]],[[-1.5,-2.5,0.5],[-3.,-1.,1.5]]],[[[-2.5,2.5,-4.],[-4.5,-3.5,2.5]],[[1.5,-0.5,-1.],[3.,-3.,-2.5]]],[[[-1.5,-0.5,-1.],[3.5,-1.,-0.5]],[[-0.5,1.,-4.5],[-3.,0.5,1.]]]]]";

  assert(f15/(2) == f35);

  // Testing "get(idx1, idx2, ...)"
  Mat g1(3,3);
  g1 = "[[1,2,3],[4,5,6],[7,8,9]]";
  int g1_exp[9] = {1,2,3,4,5,6,7,8,9};
  int k = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      assert(g1.get(i,j) == g1_exp[k]);
      k++;
    }
  }

  Mat g2(2,3);
  g2 = "[[1,2,3],[4,5,6]]";
  int g2_exp[9] = {1,2,3,4,5,6};
  k = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      assert(g2.get(i,j) == g2_exp[k]);
      k++;
    }
  }

  // Testing 1-D matrix "get()"

  Mat g11(2);
  g11 = "[-2,-9]";

  int g21[5] = {-2,-9,-9,-9,-9};

  int g31[5][1] = {{0},{1},{1},{1},{1}};
  for (int i = 0; i < 5; i++) {
  	assert(g11.get(g31[i][0]) == g21[i]);
  }

  // Testing 2-D matrix "get()"

  Mat g12(3, 2);
  g12 = "[[1,2],[-5,8],[-1,8]]";

  int g22[5] = {2,-1,8,1,-1};

  int g32[5][2] = {{0,1},{2,0},{1,1},{0,0},{2,0}};
  for (int i = 0; i < 5; i++) {
  	assert(g12.get(g32[i][0], g32[i][1]) == g22[i]);
  }

  // Testing 3-D matrix "get()"

  Mat g13(2, 2, 3);
  g13 = "[[[4,4,-3],[4,8,-4]],[[-3,-5,7],[-6,-1,5]]]";

  int g23[5] = {-5,-5,7,7,-5};

  int g33[5][3] = {{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,1}};
  for (int i = 0; i < 5; i++) {
    // std::cout << g13.get(g33[i][0], g33[i][1], g33[i][2]) << " " << g23[i] << std::endl;
  	assert(g13.get(g33[i][0], g33[i][1], g33[i][2]) == g23[i]);
  }

  // Testing 4-D matrix "get()"

  Mat g14(2, 2, 3, 3);
  g14 = "[[[[4,1,8],[8,-3,-9],[-8,-7,1]],[[-3,-9,1],[-1,-1,1],[6,-8,2]]],[[[-3,-2,7],[-1,3,4],[8,-3,1]],[[-7,2,-3],[-6,6,-3],[-6,0,-2]]]]";

  int g24[5] = {3,-3,1,-7,-1};

  int g34[5][4] = {{1,0,1,1},{0,1,0,0},{0,0,0,1},{0,0,2,1},{0,1,1,0}};
  for (int i = 0; i < 5; i++) {
  	assert(g14.get(g34[i][0], g34[i][1], g34[i][2], g34[i][3]) == g24[i]);
  }

  // Testing 5-D matrix "get()"

  Mat g15(2, 2, 2, 3, 2);
  g15 = "[[[[[-7,-7],[-7,-2],[-6,-6]],[[1,-3],[4,1],[5,-2]]],[[[0,6],[1,-3],[-6,-8]],[[-9,-5],[-9,1],[-9,-1]]]],[[[[4,8],[0,4],[-1,-9]],[[-1,2],[7,1],[-6,1]]],[[[-8,0],[-6,-4],[6,-3]],[[3,5],[-5,1],[1,-8]]]]]";

  int g25[5] = {0,-8,-3,-3,6};

  int g35[5][5] = {{0,1,0,0,0},{0,1,0,2,1},{0,1,0,1,1},{1,1,0,2,1},{1,1,0,2,0}};
  for (int i = 0; i < 5; i++) {
  	assert(g15.get(g35[i][0], g35[i][1], g35[i][2], g35[i][3], g35[i][4]) == g25[i]);
  }

  // Testing matrix multiplication

  Mat h1(8,7);
  h1 = "[[-6,3,-4,-5,4,5,-8],[8,-6,-1,3,1,7,3],[-7,1,-9,-3,-8,-4,-2],[-2,-7,6,-5,-3,-9,1],[5,-3,-6,3,-5,-6,-1],[-8,2,5,-6,-6,-8,7],[4,-8,6,8,-7,2,-3],[-9,-2,-7,6,-9,4,2]]";

  Mat h2(7,4);
  h2 = "[[3,4,7,4],[7,-6,8,-7],[1,4,5,-3],[4,-9,-3,-1],[-1,-9,7,4],[6,-7,5,-9],[8,-4,2,4]]";

  Mat h3(8,4);
  h3 = "[[-59,-52,14,-89],[58,-33,42,27],[-67,65,-157,-9],[-112,189,-89,101],[-39,78,-95,86],[-15,112,-65,21],[-11,77,-75,-12],[25,-61,-171,-71]]";
  assert(h1.mat_mul(h2) == h3);

  Mat h4(7,6);
  h4 = "[[6,1,3,-9,-1,3],[-3,-4,4,3,-3,4],[-4,6,-5,6,-9,-1],[-8,-3,-1,-1,2,-5],[1,0,1,5,-3,-6],[-3,0,-6,-3,2,3],[7,-9,-3,5,6,3]]";

  Mat h5(6,8);
  h5 = "[[3,-3,-9,-7,-9,-5,8,1],[7,0,-1,2,-6,-8,2,8],[3,3,-8,8,0,-2,-8,-7],[-5,4,3,-9,3,5,8,3],[-7,2,0,0,-7,-5,8,-1],[5,-3,5,-3,-5,-9,-4,8]]";

  Mat h6(7,8);
  h6 = "[[101,-56,-91,56,-95,-111,-66,-9],[1,15,28,6,61,33,-80,-19],[43,6,83,-51,86,66,0,98],[-82,36,55,66,98,96,-34,-70],[-28,32,-32,-26,57,87,40,-36],[-11,-26,81,-9,-11,-25,4,52],[-103,-7,0,-145,-51,11,138,-11]]";

  assert(h4.mat_mul(h5) == h6);

  Mat h7(8,6);
  h7 = "[[-9,7,-4,-9,1,-1],[7,-5,-2,-9,2,-6],[-3,2,-3,-5,0,2],[-4,-6,2,7,0,-7],[1,-4,6,7,7,5],[-6,-8,7,-8,6,-5],[2,5,2,0,-4,6],[8,-3,6,7,1,1]]";

  Mat h8(6,5);
  h8 = "[[-3,3,6,2,-4],[5,8,-6,0,3],[-8,0,-2,-4,-6],[-8,-5,6,5,-4],[7,6,-5,-4,1],[0,8,4,-8,3]]";

  Mat h9(8,5);
  h9 = "[[173,72,-151,-43,115],[56,-10,-12,17,-11],[83,48,-46,-35,62],[-90,-151,22,75,-63],[-78,18,45,-55,-58],[28,-46,-100,-64,-19],[-25,70,22,-36,9],[-136,-21,95,15,-101]]";

  assert(h7.mat_mul(h8) == h9);

  Mat h10(3,7);
  h10 = "[[-8,5,5,5,5,-4,4],[-5,6,-3,-6,-7,-2,-7],[7,-3,8,-5,8,-1,-7]]";

  Mat h11(7,4);
  h11 = "[[-2,4,-1,7],[0,3,0,4],[8,-6,8,-6],[-8,2,0,1],[8,-3,5,-9],[6,-3,-5,-7],[-8,0,0,7]]";

  Mat h12(3,4);
  h12 = "[[0,-40,93,-50],[22,31,-44,29],[204,-60,102,-130]]";

  assert(h10.mat_mul(h11) == h12);

  Mat h13(4,6);
  h13 = "[[4,6,-8,2,-5,-3],[-6,-4,5,-7,6,-8],[2,-8,-1,4,3,3],[1,-6,3,0,6,8]]";

  Mat h14(6,7);
  h14 = "[[-4,-8,0,6,-6,-2,-2],[-5,-6,-4,-3,5,1,5],[8,3,6,2,-6,4,-5],[8,-2,3,-3,6,3,-9],[-6,-7,-3,1,-5,-9,-5],[-9,5,5,-5,1,-9,6]]";

  Mat h15(4,7);
  h15 = "[[-37,-76,-66,-6,88,44,51],[64,19,-33,53,-94,25,-48],[11,15,44,10,-34,-58,-72],[-58,35,64,-4,-76,-122,-29]]";

  assert(h13.mat_mul(h14) == h15);

  Mat h16(3,5);
  h16 = "[[0,1,8,5,-3],[-4,5,-3,3,8],[7,7,8,7,4]]";

  Mat h17(5,6);
  h17 = "[[3,-5,5,-1,-4,-9],[-8,-3,6,2,-4,0],[3,-7,-8,-8,4,0],[3,-7,-3,6,-7,3],[1,-9,-8,-5,0,5]]";

  Mat h18(3,6);
  h18 = "[[28,-67,-49,-17,-7,0],[-44,-67,-39,16,-37,85],[14,-197,-40,-35,-73,-22]]";

  assert(h16.mat_mul(h17) == h18);

  Mat h19(7,5);
  h19 = "[[1,7,7,1,-1],[-8,-7,0,-8,-2],[-3,4,-6,-6,-2],[2,8,7,-1,-6],[-4,8,2,8,-1],[6,-4,-7,-7,-4],[-1,-7,3,5,-3]]";

  Mat h20(5,4);
  h20 = "[[-3,8,-5,6],[2,-2,-7,6],[-6,-2,7,8],[-3,-7,-6,-7],[-7,-5,0,-2]]";

  Mat h21(7,4);
  h21 = "[[-27,-22,-11,99],[48,16,137,-30],[85,32,-19,4],[13,23,-11,135],[-1,-103,-70,-14],[65,139,-9,13],[-23,-20,45,-53]]";

  assert(h19.mat_mul(h20) == h21);

  Mat h22(6,3);
  h22 = "[[-5,7,2],[6,-9,2],[-6,-2,-2],[1,4,-6],[-1,-9,5],[-6,4,7]]";

  Mat h23(3,5);
  h23 = "[[5,-9,4,0,-8],[5,8,-3,-1,-7],[-3,-8,7,-3,-3]]";

  Mat h24(6,5);
  h24 = "[[4,85,-27,-13,-15],[-21,-142,65,3,9],[-34,54,-32,8,68],[43,71,-50,14,-18],[-65,-103,58,-6,56],[-31,30,13,-25,-1]]";

  assert(h22.mat_mul(h23) == h24);

  Mat h25(8,8);
  h25 = "[[1,-6,5,-8,-5,2,-5,3],[-5,-2,-2,-2,1,7,-1,-4],[2,-4,-7,8,-2,-4,-1,-4],[0,3,-6,2,-1,8,8,6],[-5,-3,5,5,-7,-7,-4,-3],[1,2,4,6,-6,7,6,2],[4,-4,5,4,7,-8,2,5],[-8,-9,-5,-5,5,7,7,7]]";

  Mat h26(8,5);
  h26 = "[[2,-3,0,-8,-3],[5,0,-8,5,8],[-4,4,-6,-4,2],[-5,-5,-8,-2,-5],[-6,-8,7,-5,-1],[-4,8,-4,2,-2],[7,-5,1,4,-6],[8,8,-3,8,-3]]";

  Mat h27(8,5);
  h27 = "[[3,162,25,-9,21],[-75,38,34,15,8],[-39,-117,23,-58,-64],[107,46,-53,136,-79],[-52,6,-62,-24,30],[32,73,-158,58,-59],[-8,-102,38,-83,-72],[31,66,135,122,-115]]";

  assert(h25.mat_mul(h26) == h27);

  Mat h28(4,3);
  h28 = "[[-2,-2,3],[-2,-4,4],[2,-8,-1],[-5,-8,-9]]";

  Mat h29(3,6);
  h29 = "[[0,7,5,-3,-2,-9],[-7,0,6,-4,2,5],[3,-4,-8,6,7,-4]]";

  Mat h30(4,6);
  h30 = "[[23,-26,-46,32,21,-4],[40,-30,-66,46,24,-18],[53,18,-30,20,-27,-54],[29,1,-1,-7,-69,41]]";

  assert(h28.mat_mul(h29) == h30);

  // FROM HERE TODO

  // Testing "transposition()" method

  Mat j1(2,3);
  j1 = "[[1,2,3],[4,5,6]]";

  Mat j2(3,2);
  j2 = j1.transpose(3,2);

  // Testing "determinant()" method

  Mat k1(2,2);
  Mat k2(3,3);
  Mat k3(4,4);
  k1 = "[[4,1],[2,3]]";
  k2 = "[[2,-3,1],[2,0,-1],[1,4,5]]";
  k3 = "[[4,3,2,2],[0,1,0,-2],[1,-1,3,3],[2,3,1,1]]";
  assert(k1.determinant() == 10);
  assert(k2.determinant() == 49);
  assert(k3.determinant() == 30);

  // Testing matrix inverse method

  Mat l1(2,2);
  l1 = "[[4,7],[2,6]]";
  Mat l2(2,2);
  l2 = "[[0.6,-0.7],[-0.2,0.4]]";
  assert(l1.inverse().transpose() == l2);

  Mat l3(3,3);
  l3 = "[[1,2,3],[0,1,4],[5,6,0]]";
  Mat l4(3,3);
  l4 = "[[-24,18,5],[20,-15,-4],[-5,4,1]]";
  assert(l3.inverse() == l4);

  // Testing 2-D matrix transposition

  Mat m1(3,3);
  m1 = "[[1,2,3],[4,5,6],[7,8,9]]";
  Mat m2(3,3);
  m2 = "[[1,4,7],[2,5,8],[3,6,9]]";
  assert(m1.transpose() == m2);
  assert(m2.transpose() == m1);

  Mat m3(2,4);
  m3 = "[[1,2,3,4],[5,6,7,8]]";
  Mat m4(4,2);
  m4 = "[[1,5],[2,6],[3,7],[4,8]]";
  assert(m3.transpose() == m4);
  assert(m3.transpose().transpose() == m3);
  assert(m4.transpose() == m3);

  // Testing the outer product

  Mat n1(1,5);
  n1 = "[[0],[1],[2],[3],[4]]";
  Mat n2(3);
  n2 = "[1,2,3]";
  Mat n3(5,3);
  n3 = "[[0,0,0],[1,2,3],[2,4,6],[3,6,9],[4,8,12]]";
  assert(n1.outer(n2) == n3);

  // Testing rref

  Mat o1(3,4);
  o1 = "[[2,2,0,3],[1,1,0,2],[0,0,0,1]]";
  // print(o1.rref());
}

void tests_MP() {
  Mat a1(100,100);
  Mat a2 = a1 + a1;
  Mat a3 = a1.prod(a1);
  Mat a4 = a1.mat_mul(a1);
}
