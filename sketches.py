import numpy.random as rand
import numpy as np
import math

def sci_not(x, N=1, n=2):
    pre_sign = ""
    if x < 0:
        pre_sign = "-"

    x = abs(x)

    if x == 0:
        l = 0
        pre_sign = ""
    else:
        l = int(math.log10(x))

    post_sign = "+"
    if 0 < x < 1:
        l -= 1
        post_sign = "-"

    d = N+n-1
    x *= 10**(d-l)

    raw_len = 0
    if x != 0:
        raw_len = int(math.log10(x))
    x = round(x)

    round_len = 0
    if x != 0:
        round_len = int(math.log10(x))

    diff = round_len - raw_len
    x /= 10**(d+diff)
    l += diff

    pre_str = str(x)
    padding = N+n+1 - len(pre_str)
    pre_str += "0"*padding

    post_str = ""
    if l < 10:
        post_str += "0"
    post_str += str(abs(l))

    if l == 0:
        post_sign = "+"
    return pre_sign + pre_str + "E" + post_sign + post_str

def matrix_addition_tests(letters = ["c1", "c2", "c3", "c4"]):
    dmax = 5
    with open("addition_tests.txt", "w+") as outfile:
        for d in range(1, dmax+1):

            dims = rand.randint(2,4,d)
            vals1 = rand.randint(-9,9,tuple(dims))
            vals2 = rand.randint(-9,9,tuple(dims))

            f_dims = ""
            for i in dims:
                f_dims += "{}, ".format(i)
            f_dims = f_dims[:-2]

            vals3 = vals1 + vals2   #Operation

            m1 = str(repr(vals1))
            m2 = str(repr(vals2))
            m3 = str(repr(vals3))

            m1 = m1.replace("(", "")
            m1 = m1.replace(" ", "")
            m1 = m1.replace("\n", "")
            m1 = m1.replace(")", "")[5:]
            m2 = m2.replace("(", "")
            m2 = m2.replace(" ", "")
            m2 = m2.replace("\n", "")
            m2 = m2.replace(")", "")[5:]
            m3 = m3.replace("(", "")
            m3 = m3.replace(" ", "")
            m3 = m3.replace("\n", "")
            m3 = m3.replace(")", "")[5:]

            outfile.write("\n\n// Testing {}-D matrix addition\n".format(d))
            outfile.write("\nMat {}{}({});".format(letters[0], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[0], d, f_dims) + m1 + "\";\n")
            outfile.write("\nMat {}{}({});".format(letters[1], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[1], d, f_dims) + m2 + "\";\n")
            outfile.write("\nMat {}{}({});".format(letters[2], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[2], d, f_dims) + m3 + "\";\n")

            outfile.write("\nassert({1}{0} + {2}{0} == {3}{0});"\
            .format(d, letters[0], letters[1], letters[2]))

def matrix_subtraction_tests(letters = ["d1", "d2", "d3", "d4"]):
    dmax = 5
    with open("subtraction_tests.txt", "w+") as outfile:
        for d in range(1, dmax+1):

            dims = rand.randint(2,4,d)
            vals1 = rand.randint(-9,9,tuple(dims))
            vals2 = rand.randint(-9,9,tuple(dims))

            f_dims = ""
            for i in dims:
                f_dims += "{}, ".format(i)
            f_dims = f_dims[:-2]

            vals3 = vals1 - vals2   #Operation

            m1 = str(repr(vals1))
            m2 = str(repr(vals2))
            m3 = str(repr(vals3))

            m1 = m1.replace("(", "")
            m1 = m1.replace(" ", "")
            m1 = m1.replace("\n", "")
            m1 = m1.replace(")", "")[5:]
            m2 = m2.replace("(", "")
            m2 = m2.replace(" ", "")
            m2 = m2.replace("\n", "")
            m2 = m2.replace(")", "")[5:]
            m3 = m3.replace("(", "")
            m3 = m3.replace(" ", "")
            m3 = m3.replace("\n", "")
            m3 = m3.replace(")", "")[5:]

            outfile.write("\n\n// Testing {}-D matrix subtraction\n".format(d))
            outfile.write("\nMat {}{}({});".format(letters[0], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[0], d, f_dims) + m1 + "\";\n")
            outfile.write("\nMat {}{}({});".format(letters[1], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[1], d, f_dims) + m2 + "\";\n")
            outfile.write("\nMat {}{}({});".format(letters[2], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[2], d, f_dims) + m3 + "\";\n")

            outfile.write("\nassert({1}{0} - {2}{0} == {3}{0});"\
            .format(d, letters[0], letters[1], letters[2]))

def matrix_scalar_multiplication_tests(letters = ["e1", "e2", "e3", "e4"]):
    dmax = 5
    with open("scalar_multi_tests.txt", "w+") as outfile:
        for d in range(1, dmax+1):

            dims = rand.randint(2,4,d)
            scalar = rand.randint(-5,5)
            vals1 = rand.randint(-9,9,tuple(dims))

            f_dims = ""
            for i in dims:
                f_dims += "{}, ".format(i)
            f_dims = f_dims[:-2]

            vals2 = vals1*scalar   #Operation

            m1 = str(repr(vals1))
            m2 = str(repr(vals2))

            m1 = m1.replace("(", "")
            m1 = m1.replace(" ", "")
            m1 = m1.replace("\n", "")
            m1 = m1.replace(")", "")[5:]
            m2 = m2.replace("(", "")
            m2 = m2.replace(" ", "")
            m2 = m2.replace("\n", "")
            m2 = m2.replace(")", "")[5:]

            outfile.write("\n\n// Testing {}-D matrix-scalar multiplication\n".format(d))
            outfile.write("\nMat {}{}({});".format(letters[0], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[0], d, f_dims) + m1 + "\";\n")
            outfile.write("\nMat {}{}({});".format(letters[2], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[2], d, f_dims) + m2 + "\";\n")

            outfile.write("\nassert({1}{0}*({2}) == {3}{0});"\
            .format(d, letters[0], scalar, letters[2]))

def matrix_scalar_division_tests(letters = ["f1", "f2", "f3", "f4"]):
    dmax = 5
    with open("scalar_div_tests.txt", "w+") as outfile:
        for d in range(1, dmax+1):

            dims = rand.randint(2,4,d)
            scalar = rand.randint(-5,5)
            vals1 = rand.randint(-9,9,tuple(dims))

            if scalar == 0:
                scalar = 1
            elif scalar == 3:
                scalar = 2
            elif scalar == -3:
                scalar = -2
            elif scalar == 5:
                scalar = 4
            elif scalar == -5:
                scalar = -4

            f_dims = ""
            for i in dims:
                f_dims += "{}, ".format(i)
            f_dims = f_dims[:-2]

            vals2 = vals1/scalar   #Operation

            m1 = str(repr(vals1))
            m2 = str(repr(vals2))

            m1 = m1.replace("(", "")
            m1 = m1.replace(" ", "")
            m1 = m1.replace("\n", "")
            m1 = m1.replace(")", "")[5:]
            m2 = m2.replace("(", "")
            m2 = m2.replace(" ", "")
            m2 = m2.replace("\n", "")
            m2 = m2.replace(")", "")[5:]

            outfile.write("\n\n// Testing {}-D matrix-scalar division\n".format(d))
            outfile.write("\nMat {}{}({});".format(letters[0], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[0], d, f_dims) + m1 + "\";\n")
            outfile.write("\nMat {}{}({});".format(letters[2], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[2], d, f_dims) + m2 + "\";\n")

            outfile.write("\nassert({1}{0}/({2}) == {3}{0});"\
            .format(d, letters[0], scalar, letters[2]))

def matrix_get_tests(letters = ["g1", "g2", "g3"]):
    dmax = 5
    with open("get_tests.txt", "w+") as outfile:
        for d in range(1, dmax+1):
            N = 5
            min_size = 2
            max_size = 4
            dims = rand.randint(min_size, max_size, d, dtype = int)
            vals1 = rand.randint(-9, 9, tuple(dims), dtype = int)

            idx = np.zeros((d,N), dtype = int)
            for n,i in enumerate(dims):
                idx[n,:] = rand.randint(0, i, N)
            idx = idx.T

            vals2 = np.zeros(N, dtype = int)
            for n,i in enumerate(idx):
                vals2[n] = vals1[tuple(i)]

            idx_dims = "[{}][{}]".format(N,d)

            f_dims = ""
            for i in dims:
                f_dims += "{}, ".format(i)
            f_dims = f_dims[:-2]

            m1 = str(repr(vals1))
            m2 = str(repr(vals2))
            m3 = str(repr(idx))

            m1 = m1.replace("(", "")
            m1 = m1.replace(" ", "")
            m1 = m1.replace("\n", "")
            m1 = m1.replace(")", "")[5:]
            m2 = m2.replace("(", "")
            m2 = m2.replace(" ", "")
            m2 = m2.replace("[", "{")
            m2 = m2.replace("]", "}")
            m2 = m2.replace("\n", "")
            m2 = m2.replace(")", "")[5:]
            m3 = m3.replace("(", "")
            m3 = m3.replace(" ", "")
            m3 = m3.replace("[", "{")
            m3 = m3.replace("]", "}")
            m3 = m3.replace("\n", "")
            m3 = m3.replace(")", "")[5:]

            loop = ""
            loop += "for (int i = 0; i < {}; i++) {{\n".format(N)
            loop += "\tassert({}{}.get(".format(letters[0], d)
            for i in range(d):
                loop += "{}{}[i][{}], ".format(letters[2], d, i)
            loop = loop[:-2]
            loop += ") == {}{}[i]);\n}}".format(letters[1], d)

            outfile.write("\n\n// Testing {}-D matrix \"get()\"\n".format(d))

            outfile.write("\nMat {}{}({});".format(letters[0], d, f_dims))
            outfile.write("\n{}{} = \"".format(letters[0], d) + m1 + "\";\n")

            outfile.write("\nint {}{}[{}]".format(letters[1], d, N))
            outfile.write(" = {}".format(m2 + ";\n"))

            outfile.write("\nint {}{}{}".format(letters[2], d, idx_dims))
            outfile.write(" = {}".format(m3 + ";\n"))

            outfile.write(loop)

def matrix_multiplication_tests(letter = "h"):
    dmax = 5
    with open("multiplication_tests.txt", "w+") as outfile:
        outfile.write("// Testing matrix multiplication")
        for i in range(10):
            d = 2
            dims = rand.randint(3,9,3)
            vals1 = rand.randint(-9,9,(dims[0], dims[1]))
            vals2 = rand.randint(-9,9,(dims[1], dims[2]))

            f_dims1 = "{:d},{:d}".format(dims[0], dims[1])
            f_dims2 = "{:d},{:d}".format(dims[1], dims[2])
            f_dims3 = "{:d},{:d}".format(dims[0], dims[2])

            vals3 = np.matmul(vals1, vals2)   #Operation

            m1 = str(repr(vals1))
            m2 = str(repr(vals2))
            m3 = str(repr(vals3))

            m1 = m1.replace("(", "")
            m1 = m1.replace(" ", "")
            m1 = m1.replace("\n", "")
            m1 = m1.replace(")", "")[5:]
            m2 = m2.replace("(", "")
            m2 = m2.replace(" ", "")
            m2 = m2.replace("\n", "")
            m2 = m2.replace(")", "")[5:]
            m3 = m3.replace("(", "")
            m3 = m3.replace(" ", "")
            m3 = m3.replace("\n", "")
            m3 = m3.replace(")", "")[5:]

            outfile.write("\n\nMat {}{}({});".format(letter, 3*i+1, f_dims1))
            outfile.write("\n{}{} = \"".format(letter, 3*i+1) + m1 + "\";\n")
            outfile.write("\nMat {}{}({});".format(letter, 3*i+2, f_dims2))
            outfile.write("\n{}{} = \"".format(letter, 3*i+2) + m2 + "\";\n")
            outfile.write("\nMat {}{}({});".format(letter, 3*i+3, f_dims3))
            outfile.write("\n{}{} = \"".format(letter, 3*i+3) + m3 + "\";\n")

            outfile.write("\nassert({}{}.mat_mul({}{}) == {}{});"\
            .format(letter, 3*i+1, letter, 3*i+2, letter, 3*i+3))

def matrix_cofactor_tests(letter = "j"):
    dmax = 5
    outfile.write("\n\n// Testing 2-D matrix cofactors\n".format(d))
    with open("cofactor_tests.txt", "w+") as outfile:
        N = 10
        for i in range(N):
            dims = rand.randint(,4)
            vals1 = rand.randint(-9,9,tuple(dims))

            f_dims = ""
            for i in dims:
                f_dims += "{}, ".format(i)
            f_dims = f_dims[:-2]

            vals3 = vals1 + vals2   #Operation

            m1 = str(repr(vals1))
            m2 = str(repr(vals2))
            m3 = str(repr(vals3))

            m1 = m1.replace("(", "")
            m1 = m1.replace(" ", "")
            m1 = m1.replace("\n", "")
            m1 = m1.replace(")", "")[5:]
            m2 = m2.replace("(", "")
            m2 = m2.replace(" ", "")
            m2 = m2.replace("\n", "")
            m2 = m2.replace(")", "")[5:]
            m3 = m3.replace("(", "")
            m3 = m3.replace(" ", "")
            m3 = m3.replace("\n", "")
            m3 = m3.replace(")", "")[5:]

            outfile.write("\nMat {}{}({});".format(letter, 3*i, f_dims))
            outfile.write("\n{}{} = \"".format(letter, 3*i, f_dims) + m1 + "\";\n")
            outfile.write("\nMat {}{}({});".format(letter, 3*i + 1, f_dims))
            outfile.write("\n{}{} = \"".format(letter, 3*i + 1, f_dims) + m2 + "\";\n")
            outfile.write("\nMat {}{}({});".format(letter, 3*i + 2, f_dims))
            outfile.write("\n{}{} = \"".format(letter, 3*i + 2, f_dims) + m3 + "\";\n")

            outfile.write("\nassert({1}{0} + {2}{0} == {3}{0});"\
            .format(d, letters[0], letters[1], letters[2]))

def determinant(u):
    size = [len(u), len(u[0])]
    #if (dims != 2) {
        #std::string msg = "Can only take the determinant of a 2-D matrix";
        #throw std::domain_error(msg);
    # }

    if size[0] != size[1]:
        #std::string msg = "Can only take the determinant of a square matrix";
        #throw std::domain_error(msg);
        pass

    pm = np.zeros_like(u)
    val = 1
    for i in range(size[0]):
        val_tmp = val
        for j in range(size[1]):
            pm[i][j] = val_tmp
            val_tmp *= -1
        val *= -1

    if size[0] == 2 and size[1] == 2:
        return u[0,0]*u[1,1] - u[1,0]*u[0,1]
    else:
        total = 0
        v = np.zeros((size[0]-1, size[1]-1))
        p = 0
        for i in range(size[1]):
            p = u[0][i]
            v = v*0
            for j in range(1, size[0]):
                for k in range(size[1]):
                    if k < i:
                        v[j-1][k] = u[j][k]
                    elif k > i:
                        v[j-1][k-1] = u[j][k]
            total += determinant(v)*p*pm[0][i]
        return total

# matrix_addition_tests()
# matrix_subtraction_tests()
# matrix_scalar_multiplication_tests()
# matrix_scalar_division_tests()
# matrix_get_tests()
# matrix_multiplication_tests()

# u = np.array([[4,1],[2,3]]) #10
# u = np.array([[2,-3,1],[2,0,-1],[1,4,5]]) #49
u = np.array([[4,3,2,2],[0,1,0,-2],[1,-1,3,3],[2,3,1,1]]) #30
print(determinant(u))
