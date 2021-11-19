# Python Interior-Point Method (PYIPM)

Python Interior-Point Method (PYIPM) is a class written in python for
optimizing nonlinear programs of the form

    min f(x) subject to {ce(x) = 0} and {ci(x) >= 0}
     x

where f is a function that is being minimized with respect to unknown
weights x subject to sets of equality constraints, {ce}, and
inequality constraints, {ci}. PYIPM uses a line search interior-point
method to search for an x that is a feasible local minimizer of a
user-defined nonlinear program.

## Shift to Aesara

Originally, this module used the Theano library maintained by the
Montreal Institute for Learning Algorithms. The original version of
Theano was deprecated in 2017. Fortunately, the Aesara dev team has taken
over maintenance of Theano in a fork called
[Aesara](https://github.com/aesara-devs/aesara). The install
script (setup.py) now installs Aesara instead of Theano as a
dependency.

## Getting Started

The following instructions guide you through the necessary
prerequisites and installation of the code.

### Prerequisites

Using PYIPM requires either a python 3 or python 2.7 interpreter. In
addition to having python installed, the following packages are
required:

    NumPy, SciPy, Aesara

Aesara is used to speed up the code execution through parallelization
and compilations of certain functions used in PYIPM. Furthermore,
Aesara makes it possible to perform many of largest steps in the
algorithm on graphics processing units (GPUs). For information on
using the GPUs, see the Aesara documentation at
https://aesara.readthedocs.io/en/latest/.

All of the above required packages may be installed through pip.

Optional prerequisites include linear algebra backends that run on
CPUs:

    Intel MKL, OpenBLAS, ATLAS, or BLAS/LAPACK

and GPU backends:

    Nvidia's CUDA or OpenCL

### Installation

To install PYIPM, download pyipm.py and simply move the file to your
preferred directory. To make sure that python is able to find
pyipm.py, make sure that the folder that contains pyipm.py is on your
PYTHONPATH.

In the future, the installation will be formalized with a setup.py
file.

### Importing PYIPM

PYIPM may be imported into your own programs via

```
import pyipm
```

The entirety of the interior-point method is contained in the class
IPM (in fact this is the only content of pyipm.py except for the
example problems in the main function). Therefore, you may prefer to
just import class itself:

```
from pyipm import IPM
```

## Trying PYIPM

After satisfying the prerequisites and installing PYIPM, you can try
the code by running the example problems that appear in the 'main'
function in pyipm.py. The problems are an assortment of unconstrained
and constrained optimization problems featuring equality constraints,
inequality constraints, and both types of constraints. There are a
total of 10 example problems that may be run by choosing an integer in
1 through 10 and passing that integer as a command line argument like
so:

```
python pyipm.py 7
```

which will execute the 7th problem. If you run this exact line, you
should see something like:

```
maximize f(x,y,z) = x*y*z subject to x+y+z = 1, x >= 0, y >= 0, z >= 0

Searching for a feasible local minimizer using the exact Hessian.
OUTER ITERATION 1
* INNER ITERATION 1
* INNER ITERATION 2
* INNER ITERATION 3
OUTER ITERATION 2
* INNER ITERATION 1
* INNER ITERATION 2
* INNER ITERATION 3
Converged to Ktol tolerance after 1 outer iteration and 3 inner iterations (6 total).

Ground truth: [x, y, z] = [0.333333333333, 0.333333333333, 0.333333333333]
Solver solution: [x, y, z] = [0.333333127815, 0.333332812448, 0.333334059737]
Slack variables: s = [0.333333127815, 0.333332812448, 0.333334059737]
Lagrange multipliers: lda = [-0.111112107809, 3.44469534696e-07, 9.09227565123e-08, 2.38490989435e-06]
f(x,y,z) = 0.0370370370369
Karush-Kuhn-Tucker conditions (up to a sign):
[array([  5.8372e-07,   7.3215e-07,  -1.1461e-06]), array([  1.0917e-07,   2.4654e-08,   7.8932e-07]), array([ 0.]), array([ 0.,  0.,  0.])]
```

Note that the solver has succeeded in attaining a feasible local
minimum if you compare the 'Ground truth' line to the 'Solver
solution' line.

These example problems can also be used as a guide to construct your
own problems which may be helpful if you are unfamiliar with Aesara.

## Building and solving your own problems

The basics of writing your own problem statements are to define the
objective function, f, the equality constraints function, ce, and the
inequality constraints function, ci, as symbolic expressions that
depend on the Aesara tensor variable, x_dev. The tensor variable x_dev
is the symbolic input to all of the aforementioned functions that
represents the weights, x.

The symbolic expression for f should result in a scalar output while
the symbolic expressions for ce and ci should return one-dimensional
arrays equal to the number of equality and inequality constraints,
respectively. Once these symbolic expressions are constructed, they
need not be compiled into functions (in fact, doing so at this point
would cause pyipm.py to raise errors). You can then initialize the
class by assigning IPM to a variable, say, problem, with the following
keyword arguments:

```
problem = pyipm.IPM(x0=x0, x_dev=x_dev, f=f, ce=ce, ci=ci)
```

where the one-dimensional NumPy array x0 is a weight initialization
(which can be omitted until later).

At this point, you are now ready to solve the problem. To do this,
call the class function solve():

```
x, s, lda, fval, kkt = problem.solve(x0=x0)
```

where the weight initialization x0 must be set here if it had not been
set during initialization of the problem. The outputs of solve() are:

x = the best solution found for the weights (NumPy array)
s = the slack variables (NumPy array; empty if there are no inequality
    constraints)
lda = the Lagrange multipliers (NumPy array; empty if there are no
    constraints)
fval = the function value of the objective function at x (scalar)
kkt = the Karush-Kuhn-Tucker (KKT) conditions (length 4 list of NumPy
    arrays)

The reason why compiling f, ce, and ci would cause errors in this case
is because PYIPM uses Aesara's symbolic differentation functionality
to generate expressions for the gradients, df, dce, and dci, and, if
the exact Hessian is being used, the Hessians, d2f, d2ce, and
d2ci. Unfortunately, once these expressions are compiled, they cannot
be used for symbolic differentiation. That is not to say, however,
that you cannot use aesara.function() to precompile your expressions
into functions. You can do that at your own convenience so long as you
provide your own symbolic expressions or functions for df, dce, and
dci and, if applicable, d2f, d2ce, and d2ci. It is worth pointing out,
however, that PYIPM does much more than just automatic differentiation
(through Aesara) and compiling your expressions; PYIPM also organizes
these inputs into an overall gradient and Hessian. To take full
advantage of PYIPM for larger problems, it is best to provide PYIPM
with expressions such that it can construct and compile the symbolic
expressions into the gradient and Hessian function. Otherwise, PYIPM
will still build the gradient and Hessian using anonymous functions
which may be less efficient.

As implied in the prior paragraph, PYIPM can either use the exact
Hessian in its calculation or a quasi-Newton method known as the
Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
algorithm. Since the Hessian grows as (D+M+N)^2 where D is the number
of weights, M is the number of equality constraints, and N is the
number of inequality constraints, computing the exact Hessian may be
impractical in large scale applications. The L-BFGS algorithm allows
you to avoid this difficulty by having PYIPM replace the calculation
of the Hessian instead with a low-dimensional approximation of the
Hessian. This can be done by setting the lbfgs parameter to an integer
greater than zero that corresponds to the number of D-dimensional
vectors to store from the most recent iterations of the interior-point
algorithm. This can be applied when initializing the problem; e.g. to
store 4 iterations:

```
problem = pyipm.IPM(x0=x0, x_dev=x_dev, f=f, ce=ce, ci=ci,
lbfgs=4)
```

To use the exact Hessian, either do not set the lbfgs parameter or set
it to 0 or False.

For a more thorough guide to the parameters and public class
functions, see the docstring of class IPM in pyipm.py. For more
information about how to build and compile expressions, see the Aesara
documentation at https://aesara.readthedocs.io/en/latest/.

# Contributing, help, bug reporting, etc.

For help, comments, suggestions, bug reports, or if you are interested
in contributing, please feel free to leave a message at
https://github.com/jkaardal/pyipm.

Thank you for your interest!

# Authors

Joel T. Kaardal (https://github.com/jkaardal)

# References

[1] Nocedal J & Wright SJ, 'Numerical Optimization', 2nd Ed. Springer
(2006).

[2] Byrd RH, Nocedal J, & Schnabel RB, 'Representations of
quasi-Newton matrices and their use in limited memory methods',
Mathematical programming, 63(1), 129-156 (1994),

[3] Wachter A & Biegler LT, 'On the implementation of an
interior-point filter line-search algorithm for large-scale nonlinear
programming', Mathematical programming, 106(1), 25-57 (2006).

# License

This project is licensed under the MIT License (see
[LICENSE.md](LICENSE.md) for details).