from __future__ import print_function

import aesara as theano
import aesara.tensor as T
import numpy as np
from aesara.tensor.nlinalg import pinv
from aesara.tensor.basic import diag
from aesara.tensor.nlinalg import eigh
from aesara.tensor.slinalg import eigvalsh
from aesara.ifelse import ifelse

try:
    FunctionType = theano.compile.function_module.Function
except AttributeError:
    FunctionType = theano.compile.function.types.Function


def sym_solve(A, b):
    # TODO: update to sym from gen solve once that becomes functional.
    return T.slinalg.solve(A, b, assume_a='gen')


class IPM:
    """Solve nonlinear, nonconvex optimization problems using an interior-point method.


       Detailed Description:
         This solver uses a line search primal-dual interior-point method to solve
         problems of the form

           min f(x)   subject to {ce(x) = 0} and {ci(x) >= 0}
            x

         where f(x) is a function of the weights x, {ce(x) = 0} is the set of M 
         equality constraints, and {ci(x) <= 0} is the set of N inequality
         constraints. The solver finds a solution to an 'unconstrained'
         transformation of the problem by forming the Lagrangian augmented by a
         barrier term with coefficient mu when N > 0:

           L(x, s, lda) = f - ce.dot(lda[:M]) - (ci-s).dot(lda[M:]) - mu * sum(log(s))

         where s >= 0 are the slack variables (only used when N > 0) that transform
         the inequality constraints into equality constraints and lda are the
         Lagrange multipliers where lda[M:] >= 0. The optimization completes when
         the first-order Karush-Kuhn-Tucker (KKT) conditions are satisfied to the
         desired precision.

         For more details on this algorithm, consult the references at the bottom.
         In particular, this algorithm uses a line search interior-point 
         algorithm with a merit function based on Ch. 19 of [1].


       Dependencies:
           Required:
             NumPy
             SciPy
             Theano
           Optional:
             Intel MKL, OpenBlas, ATLAS, or BLAS/LAPACK
             Nvidia's CUDA or OpenCL
               for more details on support for GPU software, see the Theano
               documentation.


       Examples:
           For example usage of this solver class, see the problem definitions in
           the main function at the bottom of this file, pyipm.py. Each of these
           example problems can be run from the command line using argv input. For
           example, if one wants to run example problem 3, enter this command into
           the terminal from the directory that contains pyipm.py

               python pyipm.py 3

           and the solver will print the solution to screen. There are 10 example
           problems that are called by numbers on the range 1 through 10.


       Input types:
           symbolic expression: refers to equations constructed using Theano
             objects and syntax.
           symbolic scalar/array: is the output of a symbolic expression.

       Class Variables:
           x0 (NumPy array): weight initialization (size D).
           x_dev (Theano tensor): symbolic weight variables.
           f (symbolic expression): objective/cost function; i.e. the function to be
                 minimized.
               [Args] x_dev
               [Returns] symbolic scalar
               [NOTE] see note 1) below.
           df (symbolic expression, OPTIONAL): gradient of the objective
                 function with respect to (wrt) x_dev.
               [Args] x_dev
               [Default] df is assigned through automatic symbolic differentiation
                 of f wrt x_dev
               [Returns] symbolic array (size D)
               [NOTE] see note 1) below.
           d2f (symbolic expression, OPTIONAL): hessian of the objective function wrt
                 x_dev.
               [Args] x_dev
               [Default] d2f is assigned through automatic symbolic differentiation
                 of f wrt x_dev
               [Returns] symbolic array (size DxD)
               [NOTE] see notes 1) and 3) below.
           ce (Theano expression, OPTIONAL): symbolic expression for the equality
                 constraints as a function of x_dev. This is required if dce or
                 d2ce is not None.
               [Args] x_dev
               [Default] None
               [Returns] symbolic array (size M)
               [NOTE] see note 1) below.
           dce (Theano expression, OPTIONAL): symbolic expression for the Jacobian
                 of the equality constraints wrt x_dev.
               [Args] x_dev
               [Default] if ce is not None, then dce is assigned through automatic
                 symbolic differentiation of ce wrt x_dev; otherwise None.
               [Returns] symbolic array (size DxM)
               [NOTE] see notes 1) and 2) below.
           d2ce (Theano expression, OPTIONAL): symbolic expression for the Hessian
                 of the equality constraints wrt x_dev and lambda_dev (see below).
               [Args] x_dev, lambda_dev
               [Default] if ce is not None, then d2ce is assigned through automatic
                 symbolic differentiation of ce wrt x_dev; otherwise None.
               [Returns] symbolic array (size DxD)
               [NOTE] see notes 1) and 3) below.
           ci (Theano expression, OPTIONAL): symbolic expression for the
                 inequality constraints as a function of x_dev. Required if dci or
                 d2ci are not None.
               [Args] x_dev
               [Default] None
               [Returns] symbolic array (size N)
               [NOTE] see note 1) below.
           dci (Theano expression, OPTIONAL): symbolic expression for the Jacobian 
                 of the inequality constraints wrt x_dev.
               [Args] x_dev
               [Default] if ci is not None, then dci is assigned through automatic
                 symbolic differentiation of ci wrt x_dev; otherwise None
               [Returns] symbolic array (size DxN)
               [NOTE] see notes 1) and 2) below.
           d2ci (Theano expression, OPTIONAL): symbolic expression for the Hessian
                 of the inequality constraints wrt x_dev and lambda_dev (see below).
               [Args] x_dev, lambda_dev
               [Default] if ci is not None, then d2ci is assigned through autormatic
                 symbolic differentiation of ci wrt x_dev; otherwise None
               [Returns] symbolic array (size DxD)
               [NOTE] see notes 1) and 3) below.
           lda0 (NumPy array, OPTIONAL): Lagrange multiplier initialization (size
                 M+N). For equality constraints, lda0[:M] may take on any sign while
                 for inequality constraints all elements of lda0[M:] must be >=0.
               [Default] if ce or ci is not None, then lda0 is initialized using dce
                 (if ce is not None), dci (if ci is not None), and df all evaluated
                 at x0 and the Moore-Penrose pseudoinverse; otherwise None
           lambda_dev (Theano expression, OPTIONAL) symbolic Lagrange multipliers.
                 This only required if you supply your own input for d2ce or d2ci.
               [Default] None
           s0 (NumPy array, OPTIONAL): slack variables initialization (size N).
                 These are only set when inequality constraints are in use. The
                 slack varables must be s0 >= 0.
               [Default] if ci is not None, then s0 is set to the larger of ci
                 evaluated at x0 or Ktol; otherwise None
           mu (float, OPTIONAL): barrier parameter initialization (scalar>0).
               [Default] 0.2
           nu (float, OPTIONAL): merit function parameter initialization (scalar>0).
               [Default] 10.0
           rho (float, OPTIONAL): factor used for testing and updating nu (scalar
                 in (0,1)).
               [Default] 0.1.
           tau (float, OPTIONAL): fraction-to-the-boundary rule and backtracking
                 line search parameter (scalar in (0,1)).
               [Default] 0.995
           eta (float, OPTIONAL): Armijo rule parameter (Wolfe conditions) (scalar
                 in (0,1)).
               [Default] 1.0E-4
           beta (float, OPTIONAL): power factor used in Hessian regularization to
                 combat ill-conditioning. This is only relevant if ce is not None.
               [Default] 0.4
           miter (int, OPTIONAL): number of 'inner' iterations where mu is held
                 constant.
               [Default] 20
           niter (int, OPTIONAL): number of 'outer' iterations where mu is
                 adjusted.
               [Default] 10
           Xtol (float, OPTIONAL): weight precision tolerance (used only in
                 fraction-to-the-boundary rule).
               [Default] np.finfo(float_dtype).eps (machine precision of
                 float_dtype; see below)
           Ktol (float, OPTIONAL): convergence tolerance on the Karush-Kuhn-
                 Tucker (KKT) conditions.
               [Default] 1.0E-4
           Ftol (float, OPTIONAL): convergence tolerance on the change in f
                 between iterations. For constrained problems, f is measured after
                 each outer iteration and compared to the prior outer iteration.
               [Default] None (when set to None, convergence is determined via the
                 KKT conditions alone).
           lbfgs (integer, OPTIONAL): solve using the L-BFGS approximation of the
                 Hessian; set lbfgs to a positive integer equal to the number of
                 past iterations to store. If lbfgs=0 or False, the exact Hessian
                 will be used.
               [Default] False
           lbfgs_zeta (float, OPTIONAL): initialize the scaling of the initial
                 Hessian approximation with respect to the weights. The initial
                 approximation is lbfgs_zeta multiplied by a DxD identity matrix.
                 This must be a positive scalar.
               [Default] 1.0
           float_dtype (dtype, OPTIONAL): set the universal precision of all float
                 variables.
               [Default] np.float64 (64-bit floats)
               [NOTE] Using 32-bit precision is not recommended; the numerical
                 inaccuracy can lead to slow convergence.
           verbosity (integer, OPTIONAL): screen output level from -1 to 3 where -1
                 is no feedback and 3 is maximum feedback.
               [Default] 1

           Notes:
           -----------------
           1) For flexibility, symbolic expressions for f, df, d2f, ce, dce, d2ce, ci
           dci, and d2ci may be replaced with functions. Keep in mind, however, that
           replacing f, ce, or ci with functions will disable automatic
           differentiation capabilities. This option is available for those who wish 
           to avoid redefining Theano functions that have already been compiled.
           However, using symbolic expressions may lead to a quicker optimization.

           2) When defining your own Jacobians, dce and dci, keep in mind that dce and
           dci are transposed relative to the output of Theano.gradient.jacobian()
           and should be size DxM or DxN, respectively.

           3) Depending on the sophistication of the problem you wish to optimize, it
           may be better write your own symbolic expression/function for the second
           derivative matrices d2f, d2ce, or/and d2ci. On some complicated and large
           problems, automatic differentiation of the second derivatives may be
           inefficient leading to slow computations of the Hessian.


       Class Functions:
           compile(nvar=None, neq=None, nineq=None): validate input,
                 form expressions for the Lagrangian and its gradient and Hessian,
                 form expressions for weight and Lagrange multiplier initialization,
                 define device variables, and compile symbolic expressions.
               [Args] nvar (optional), neq (optional), nineq (optional)
                 * nvar (int, scalar) must be set to the number of weights if x0 is
                   uninitialized
                 * neq (int, scalar) must be set to the number of equality constraints
                   if x0 is unintialized, M
                 * nineq (int, scalar) must be set to the number of inequality
                   constraints if x0 is uninitialized, N (scalar)
               [NOTE] If the user runs compile on an instance of the
                 solver object and then intends to reuse the solver object after
                 changing the symbolic Theano expressions defined in the Class 
                 Variables section, compile will need to be rerun to
                 compile any modified/new expressions.
           solve(x0=None, s0=None, lda0=None, force_recompile=False): run the interior
                 -point method solver.
               [Args] x0 (optional), s0 (optional), lda0 (optional),
                  force_recompile (optional)
                 * x0 (NumPy array, size D) can be used to initialize the weights if
                   they are not already initialized or to reinitialize the weights
                 * s0 (NumPy array, size N) gives the user control over initialization
                   of the slack variables, if desired (size N)
                 * lda0 (NumPy array, size M+N) gives the user control over
                   initialization of the Lagrange multipliers, if desired
                 * force_recompile (bool, scalar) the solve() class function
                   automatically calls the compile() class function on its first
                   execution. On subsequent executions, solve() will not automatically
                   recompile unless force_recompile is set to True.
               [Returns] (x, s, lda, fval, kkt)
                 * x (NumPy array, size D) are the weights at the solution
                 * s (NumPy array, size N) are the slack variables at the solution
                 * lda (NumPy array, size M+N) are the Lagrange multipliers at the
                   solution
                 * fval (float, scalar) is f evaluated at x
                 * kkt (list) is a list of the first-order KKT conditions solved at x
                   (see class function KKT for details)
               [NOTE] If the solver is used more than once, it will likely be
                 necessary to reinitialize mu and nu since they are left in their
                 final state after the solver is used.
           KKT(x, s, lda): calculate the first-order KKT conditions.
               [Args] x, s, lda
                 * x (NumPy array, size D) are the weights
                 * s (NumPy array, size N) are the slack variables
                 * lda (NumPy array, size M+N) are the Lagrange multipliers
               [Returns] [kkt1, kkt2, kkt3, kkt4]
                 * kkt1 (NumPy array, size D) is the gradient of the Lagrangian barrier
                   problem at x, s, and lda
                 * kkt2 (NumPy array, size N) is the gradient of the dual barrier
                   problem at x, s, and lda; if there are no inequality constraints,
                   then kkt2=0
                 * kkt3 (NumPy array, size M) is the equality constraint satisfaction
                   (i.e. solve ce at x); if there are no equality constraints, then
                   kkt3=0
                 * kkt4 (NumPy array, size N) is the inequality constraint 
                   satisfaction (i.e. solve ci at x); if there are inequality
                   constraints, then kkt=0
               [NOTE] All arguments are required. If s and/or lda are irrelevant to
                 the user's problem, set those variables to a 0 dimensional NumPy
                 array.


       References:
           [1] Nocedal J & Wright SJ, 'Numerical Optimization', 2nd Ed. Springer (2006).
           [2] Byrd RH, Nocedal J, & Schnabel RB, 'Representations of quasi-Newton
                 matrices and their use in limited memory methods', Mathematical
                 programming, 63(1), 129-156 (1994).
           [3] Wachter A & Biegler LT, 'On the implementation of an interior-point
                 filter line-search algorithm for large-scale nonlinear programming',
                 Mathematical programming, 106(1), 25-57 (2006).


           TO DO: Translate line-search into Theano
    """

    def __init__(self, x0=None, x_dev=None, f=None, df=None, d2f=None, ce=None, dce=None, d2ce=None, ci=None, dci=None,
                 d2ci=None, lda0=None, lambda_dev=None, s0=None, mu=0.2, nu=10.0, rho=0.1, tau=0.995, eta=1.0E-4,
                 beta=0.4, miter=20, niter=10, Xtol=None, Ktol=1.0E-4, Ftol=None, lbfgs=False, lbfgs_zeta=None,
                 float_dtype=np.float64, verbosity=1):

        self.x0 = x0
        self.x_dev = x_dev
        self.lda0 = lda0
        self.lambda_dev = lambda_dev
        self.s0 = s0

        self.f = f
        self.df = df
        self.d2f = d2f
        self.ce = ce
        self.dce = dce
        self.d2ce = d2ce
        self.ci = ci
        self.dci = dci
        self.d2ci = d2ci

        self.nvar = None
        self.neq = None
        self.nineq = None

        self.eps = np.finfo(float_dtype).eps

        self.mu = mu
        self.nu = nu
        self.rho = rho
        self.tau = tau
        self.eta = eta
        self.beta = beta
        self.miter = miter
        self.niter = niter
        if Xtol:
            self.Xtol = Xtol
        else:
            self.Xtol = self.eps
        self.Ktol = Ktol
        self.Ftol = Ftol

        self.reg_coef = float_dtype(np.sqrt(self.eps))

        self.lbfgs = lbfgs
        if self.lbfgs and lbfgs_zeta is None:
            self.lbfgs_zeta = float_dtype(1.0)
        else:
            self.lbfgs_zeta = lbfgs_zeta
        self.lbfgs_fail_max = lbfgs

        self.float_dtype = float_dtype
        self.nu_dev = theano.shared(self.float_dtype(self.nu), name='nu_dev')
        self.mu_dev = theano.shared(self.float_dtype(self.mu), name='mu_dev')
        self.dz_dev = T.vector('dz_dev')
        self.b_dev = T.matrix('b_dev')
        self.M_dev = T.matrix('M_dev')
        self.s_dev = T.vector('s_dev')

        self.verbosity = verbosity

        self.delta0 = self.reg_coef

        self.numpy_printoptions = np.set_printoptions(precision=4)

        self.compiled = False

    @staticmethod
    def check_precompile(func):
        """Check if the Theano expression is actually a Theano function. If so,
           return True, otherwise return False.
        """
        return isinstance(func, FunctionType)

    def validate(self):
        """Validate inputs
        """
        assert self.x_dev is not None
        assert self.f is not None
        assert (self.ce is not None) or (
            self.ce is None and self.dce is None and self.d2ce is None)
        assert (self.ci is not None) or (
            self.ci is None and self.dci is None and self.d2ci is None)
        assert self.mu > 0.0
        assert self.nu > 0.0
        assert 0.0 < self.eta < 1.0
        assert 0.0 < self.rho < 1.0
        assert 0.0 < self.tau < 1.0
        assert self.beta < 1.0
        assert self.miter >= 0 and isinstance(self.miter, int)
        assert self.niter >= 0 and isinstance(self.miter, int)
        assert self.Xtol >= self.eps
        assert self.Ktol >= self.eps
        assert self.Ftol is None or self.Ftol >= 0.0
        assert self.lbfgs >= 0 or self.lbfgs == False
        if self.lbfgs:
            assert isinstance(self.lbfgs, int)
        assert self.lbfgs_zeta is None or self.lbfgs_zeta > 0.0

    def compile(self, nvar=None, neq=None, nineq=None):
        """Validate some of the input variables and compile the objective function,
           the gradient, and the Hessian with constraints.
        """
        # get number of variables and constraints
        if nvar is not None:
            self.nvar = nvar
        if neq is not None:
            self.neq = neq
        else:
            self.neq = None
        if nineq is not None:
            self.nineq = nineq
        else:
            self.nineq = None

        # check if any functions are precompiled
        f_precompile = self.check_precompile(self.f)
        df_precompile = self.check_precompile(self.df)
        d2f_precompile = self.check_precompile(self.d2f)
        ce_precompile = self.check_precompile(self.ce)
        dce_precompile = self.check_precompile(self.dce)
        d2ce_precompile = self.check_precompile(self.d2ce)
        ci_precompile = self.check_precompile(self.ci)
        dci_precompile = self.check_precompile(self.dci)
        d2ci_precompile = self.check_precompile(self.d2ci)
        if any([f_precompile, df_precompile, d2f_precompile, ce_precompile, dce_precompile, d2ce_precompile,
                ci_precompile, dci_precompile, d2ci_precompile]):
            precompile = True
        else:
            precompile = False

        # get the number of equality and inequality constraints if not provided by the user
        if self.ce is not None and self.neq is None:
            if ce_precompile:
                CE = self.ce
            else:
                CE = theano.function(inputs=[self.x_dev], outputs=self.ce)

            c = CE(self.x0)
            self.neq = c.size
        elif neq is None:
            self.neq = 0
        else:
            self.neq = neq

        if self.ci is not None and self.nineq is None:
            if ci_precompile:
                CI = self.ci
            else:
                CI = theano.function(inputs=[self.x_dev], outputs=self.ci)

            c = CI(self.x0)
            self.nineq = c.size
        elif nineq is None:
            self.nineq = 0
        else:
            self.nineq = nineq

        # declare device variables
        if self.lambda_dev is None:
            self.lambda_dev = T.vector('lamda_dev')

        # use automatic differentiation if gradient and/or Hessian (if applicable) of f expressions are not provided
        if self.df is None:
            df = T.grad(self.f, self.x_dev)
        else:
            df = self.df
        if not self.lbfgs and self.d2f is None:
            d2f = theano.gradient.hessian(cost=self.f, wrt=self.x_dev)
        else:
            d2f = self.d2f

        # construct expression for the constraint Jacobians and Hessians (if exact Hessian is used)
        if self.neq:
            if self.dce is None:
                dce = theano.gradient.jacobian(
                    self.ce, wrt=self.x_dev).reshape((self.neq, self.nvar)).T
            else:
                dce = self.dce
            if not self.lbfgs:
                if self.d2ce is None:
                    d2ce = theano.gradient.hessian(cost=T.sum(
                        self.ce * self.lambda_dev[:self.neq]), wrt=self.x_dev)
                else:
                    d2ce = self.d2ce

        if self.nineq:
            Sigma = diag(self.lambda_dev[self.neq:] / (self.s_dev + self.eps))
            if self.dci is None:
                dci = theano.gradient.jacobian(
                    self.ci, wrt=self.x_dev).reshape((self.nineq, self.nvar)).T
            else:
                dci = self.dci
            if not self.lbfgs:
                if self.d2ci is None:
                    d2ci = theano.gradient.hessian(cost=T.sum(
                        self.ci * self.lambda_dev[self.neq:]), wrt=self.x_dev)
                else:
                    d2ci = self.d2ci

        # if some expressions have been precompiled into functions, compile any remaining expressions
        if precompile:
            if not f_precompile:
                f_func = theano.function(inputs=[self.x_dev], outputs=self.f)
            else:
                f_func = self.f
            if not df_precompile:
                df_func = theano.function(inputs=[self.x_dev], outputs=df)
            else:
                df_func = df
            if not self.lbfgs:
                if not d2f_precompile:
                    d2f_func = theano.function(
                        inputs=[self.x_dev], outputs=d2f)
                else:
                    d2f_func = d2f
            if self.neq:
                if not ce_precompile:
                    ce_func = theano.function(
                        inputs=[self.x_dev], outputs=self.ce)
                else:
                    ce_func = self.ce
                if not dce_precompile:
                    dce_func = theano.function(
                        inputs=[self.x_dev], outputs=dce.reshape((self.nvar, self.neq)))
                else:
                    def dce_func(x): return dce(
                        x).reshape((self.nvar, self.neq))
                if not self.lbfgs:
                    if not d2ce_precompile:
                        d2ce_func = theano.function(
                            inputs=[self.x_dev, self.lambda_dev], outputs=d2ce)
                    else:
                        d2ce_func = d2ce
            if self.nineq:
                if not ci_precompile:
                    ci_func = theano.function(
                        inputs=[self.x_dev, self.s_dev], outputs=self.ci - self.s_dev)
                else:
                    def ci_func(x, s): return self.ci(x) - s
                if not dci_precompile:
                    dci_func = theano.function(
                        inputs=[self.x_dev], outputs=dci.reshape((self.nvar, self.nineq)))
                else:
                    def dci_func(x): return dci(
                        x).reshape((self.nvar, self.nineq))
                if not self.lbfgs:
                    if not d2ci_precompile:
                        d2ci_func = theano.function(
                            inputs=[self.x_dev, self.lambda_dev], outputs=d2ci)
                    else:
                        d2ci_func = d2ci

        # construct composite expression for the constraints
        if self.neq or self.nineq:
            if precompile:
                if self.neq and self.nineq:
                    def con(x, s): return np.concatenate([ce_func(x).reshape((self.neq,)),
                                                          ci_func(x, s).reshape((self.nineq,))], axis=0)
                elif self.neq:
                    def con(x, s): return ce_func(x).reshape((self.neq,))
                else:
                    def con(x, s): return ci_func(x, s).reshape((self.nineq,))
            else:
                con = T.zeros((self.neq + self.nineq,))
                if self.neq:
                    con = T.set_subtensor(con[:self.neq], self.ce)
                if self.nineq:
                    con = T.set_subtensor(con[self.neq:], self.ci - self.s_dev)

        # construct composite expression for the constraints Jacobian
        if self.neq or self.nineq:
            if precompile:
                if self.neq and self.nineq:
                    def jaco_top(x): return np.concatenate([dce_func(x).reshape((self.nvar, self.neq)),
                                                            dci_func(x).reshape((self.nvar, self.nineq))], axis=1)
                    jaco_bottom = np.concatenate(
                        [np.zeros((self.nineq, self.neq)), -np.eye(self.nineq)], axis=1)

                    def jaco(x): return np.concatenate(
                        [jaco_top(x), jaco_bottom], axis=0)
                elif self.neq:
                    def jaco(x): return dce_func(
                        x).reshape((self.nvar, self.neq))
                else:
                    def jaco(x): return np.concatenate([
                        dci_func(x).reshape((self.nvar, self.nineq)),
                        -np.eye(self.nineq)
                    ], axis=0)
            else:
                jaco = T.zeros((self.nvar + self.nineq, self.neq + self.nineq))
                if self.neq:
                    jaco = T.set_subtensor(jaco[:self.nvar, :self.neq], dce)
                if self.nineq:
                    jaco = T.set_subtensor(jaco[:self.nvar, self.neq:], dci)
                    jaco = T.set_subtensor(
                        jaco[self.nvar:, self.neq:], -T.eye(self.nineq))

        # construct expression for the gradient
        if precompile:
            if self.neq and self.nineq:
                def grad_x(x, lda): return (df_func(x) - np.dot(dce_func(x), lda[:self.neq]) -
                                            np.dot(dci_func(x), lda[self.neq:]))
            elif self.neq:
                def grad_x(x, lda): return df_func(
                    x) - np.dot(dce_func(x), lda)
            elif self.nineq:
                def grad_x(x, lda): return df_func(
                    x) - np.dot(dci_func(x), lda)
            else:
                def grad_x(x, lda): return df_func(x)

            if self.nineq:
                grad_s = self.lambda_dev[self.neq:] - \
                    self.mu_dev / (self.s_dev + self.eps)
                grad_s = theano.function(inputs=[self.x_dev, self.s_dev, self.lambda_dev], outputs=grad_s,
                                         on_unused_input='ignore')

            if self.neq:
                if ce_precompile:
                    def grad_lda_eq(x): return ce_func(x).ravel()
                else:
                    grad_lda_eq = theano.function(inputs=[self.x_dev], outputs=self.ce.ravel(),
                                                  on_unused_input='ignore')

            if self.nineq:
                if ci_precompile:
                    def grad_lda_ineq(x, s): return ci_func(x, s).ravel()
                else:
                    grad_lda_ineq = theano.function(inputs=[self.x_dev, self.s_dev],
                                                    outputs=(self.ci - self.s_dev).ravel(), on_unused_input='ignore')

            if self.neq and self.nineq:
                def grad(x, s, lda): return np.concatenate([grad_x(x, lda), grad_s(x, s, lda), grad_lda_eq(x),
                                                            grad_lda_ineq(x, s)], axis=0)
            elif self.neq:
                def grad(x, s, lda): return np.concatenate(
                    [grad_x(x, lda), grad_lda_eq(x)], axis=0)
            elif self.nineq:
                def grad(x, s, lda): return np.concatenate([grad_x(x, lda), grad_s(x, s, lda), grad_lda_ineq(x, s)],
                                                           axis=0)
            else:
                def grad(x, s, lda): return grad_x(x, lda)
        else:
            grad = T.zeros((self.nvar + 2 * self.nineq + self.neq,))
            grad = T.set_subtensor(grad[:self.nvar], df)
            if self.neq:
                grad = T.inc_subtensor(
                    grad[:self.nvar], -T.dot(dce, self.lambda_dev[:self.neq]))
                grad = T.set_subtensor(
                    grad[self.nvar + self.nineq:self.nvar + self.nineq + self.neq], self.ce)
            if self.nineq:
                grad = T.inc_subtensor(
                    grad[:self.nvar], -T.dot(dci, self.lambda_dev[self.neq:]))
                grad = T.set_subtensor(grad[self.nvar:self.nvar + self.nineq], self.lambda_dev[self.neq:] -
                                       self.mu_dev / (self.s_dev + self.eps))
                grad = T.set_subtensor(
                    grad[self.nvar + self.nineq + self.neq:], self.ci - self.s_dev)

        # construct expressions for the merit function
        if precompile:
            if self.nineq:
                bar_func = self.mu_dev * T.sum(T.log(self.s_dev))
                bar_func = theano.function(
                    inputs=[self.s_dev], outputs=bar_func, on_unused_input='ignore')
            if self.neq and self.nineq:
                def phi(x, s): return (f_func(x) + self.nu_dev.get_value() *
                                       (np.sum(np.abs(ce_func(x))) + np.sum(np.abs(ci_func(x, s)))) - bar_func(s))
            elif self.neq:
                def phi(x, s): return f_func(x) + \
                    self.nu_dev.get_value() * np.sum(np.abs(ce_func(x)))
            elif self.nineq:
                def phi(x, s): return f_func(x) + self.nu_dev.get_value() * \
                    np.sum(np.abs(ci_func(x, s))) - bar_func(s)
            else:
                def phi(x, s): return f_func(x)
        else:
            phi = self.f
            if self.neq:
                phi += self.nu_dev * T.sum(T.abs_(self.ce))
            if self.nineq:
                phi += self.nu_dev * \
                    T.sum(T.abs_(self.ci - self.s_dev)) - \
                    self.mu_dev * T.sum(T.log(self.s_dev))

        # construct expressions for the merit function gradient
        if precompile:
            if self.nineq:
                dbar_func = T.dot(
                    self.mu_dev / (self.s_dev + self.eps), self.dz_dev[self.nvar:])
                dbar_func = theano.function(inputs=[self.s_dev, self.dz_dev], outputs=dbar_func,
                                            on_unused_input='ignore')
            if self.neq and self.nineq:
                def dphi(x, s, dz): return (np.dot(df_func(x), dz[:self.nvar]) - self.nu_dev.get_value() *
                                            (np.sum(np.abs(ce_func(x))) + np.sum(np.abs(ci_func(x, s)))) -
                                            dbar_func(s, dz))
            elif self.neq:
                def dphi(x, s, dz): return (np.dot(df_func(x), dz[:self.nvar]) -
                                            self.nu_dev.get_value() * np.sum(np.abs(ce_func(x))))
            elif self.nineq:
                def dphi(x, s, dz): return (np.dot(df_func(x), dz[:self.nvar]) - self.nu_dev.get_value() *
                                            np.sum(np.abs(ci_func(x, s))) - dbar_func(s, dz))
            else:
                def dphi(x, s, dz): return np.dot(df_func(x), dz[:self.nvar])
        else:
            dphi = T.dot(df, self.dz_dev[:self.nvar])
            if self.neq:
                dphi -= self.nu_dev * T.sum(T.abs_(self.ce))
            if self.nineq:
                dphi -= (self.nu_dev * T.sum(T.abs_(self.ci - self.s_dev)) +
                         T.dot(self.mu_dev / (self.s_dev + self.eps), self.dz_dev[self.nvar:]))

        # construct expression for initializing the Lagrange multipliers
        if self.neq or self.nineq:
            if precompile:
                def init_lambda(x): return np.dot(np.linalg.pinv(jaco(x)[:self.nvar, :]),
                                                  df_func(x).reshape((self.nvar, 1))).reshape((self.neq + self.nineq,))
            else:
                init_lambda = T.dot(pinv(jaco[:self.nvar, :]),
                                    df.reshape((self.nvar, 1))).reshape((self.neq + self.nineq,))

        # construct expression for initializing the slack variables
        if self.nineq:
            if precompile:
                def init_slack(x): return np.max(np.concatenate([
                    ci_func(x, np.zeros((self.nineq,))
                            ).reshape((self.nineq, 1)),
                    self.Ktol * np.ones((self.nineq, 1))
                ], axis=1), axis=1)
            else:
                init_slack = T.max(T.concatenate([
                    self.ci.reshape((self.nineq, 1)),
                    self.Ktol * T.ones((self.nineq, 1))
                ], axis=1), axis=1)

        # construct expression for gradient of f( + the barrier function)
        if precompile:
            if self.nineq:
                dbar_func2 = -self.mu_dev / (self.s_dev + self.eps)
                dbar_func2 = theano.function(
                    inputs=[self.s_dev], outputs=dbar_func2, on_unused_input='ignore')

                def barrier_cost_grad(x, s): return np.concatenate(
                    [df_func(x), dbar_func2(s)], axis=0)
            else:
                def barrier_cost_grad(x, s): return df_func(x)
        else:
            barrier_cost_grad = T.zeros((self.nvar + self.nineq,))
            barrier_cost_grad = T.set_subtensor(
                barrier_cost_grad[:self.nvar], df)
            if self.nineq:
                barrier_cost_grad = T.set_subtensor(barrier_cost_grad[self.nvar:],
                                                    -self.mu_dev / (self.s_dev + self.eps))

        # construct expression for the Hessian of the Lagrangian (assumes Lagrange multipliers included in
        # d2ce/d2ci expressions), if applicable
        if not self.lbfgs:
            if precompile:
                # construct expression for the Hessian of the Lagrangian
                if self.neq and self.nineq:
                    def d2L(x, lda): return d2f_func(x) - \
                        d2ce_func(x, lda) - d2ci_func(x, lda)
                elif self.neq:
                    def d2L(x, lda): return d2f_func(x) - d2ce_func(x, lda)
                elif self.nineq:
                    def d2L(x, lda): return d2f_func(x) - d2ci_func(x, lda)
                else:
                    def d2L(x, lda): return d2f_func(x)

                # construct expression for the symmetric Hessian matrix
                if self.neq or self.nineq:
                    if self.nineq:
                        def hess_upper_left(x, s, lda): return np.concatenate([
                            np.concatenate([
                                np.triu(d2L(x, lda)),
                                np.zeros((self.nvar, self.nineq))
                            ], axis=1),
                            np.concatenate([
                                np.zeros((self.nineq, self.nvar)),
                                np.diag(lda[self.neq:] / (s + self.eps))
                            ], axis=1)
                        ], axis=0)
                    else:
                        def hess_upper_left(
                            x, s, lda): return np.triu(d2L(x, lda))

                    def hess_upper_right(x): return jaco(x)

                    def hess_upper(x, s, lda): return np.concatenate([
                        hess_upper_left(x, s, lda),
                        hess_upper_right(x)
                    ], axis=1)

                    def hess_triu(x, s, lda): return np.concatenate([
                        hess_upper(x, s, lda),
                        np.zeros((self.neq + self.nineq, self.nvar +
                                  2 * self.nineq + self.neq))
                    ], axis=0)
                    def hess(x, s, lda): return hess_triu(
                        x, s, lda) + np.triu(hess_triu(x, s, lda), k=1).T
                else:
                    def hess_triu(x, s, lda): return np.triu(d2L(x, lda))
                    def hess(x, s, lda): return hess_triu(
                        x, s, lda) + np.triu(hess_triu(x, s, lda), k=1).T
            else:
                # construct expression for the Hessian of the Lagrangian
                d2L = d2f
                if self.neq:
                    d2L -= d2ce
                if self.nineq:
                    d2L -= d2ci

                # construct expression for the symmetric Hessian matrix
                hess = T.zeros((self.nvar + 2 * self.nineq +
                                self.neq, self.nvar + 2 * self.nineq + self.neq))
                hess = T.set_subtensor(
                    hess[:self.nvar, :self.nvar], T.triu(d2L))
                if self.neq:
                    hess = T.set_subtensor(
                        hess[:self.nvar, (self.nvar + self.nineq):(self.nvar + self.nineq + self.neq)],
                        dce
                    )
                if self.nineq:
                    hess = T.set_subtensor(
                        hess[:self.nvar, (self.nvar + self.nineq + self.neq):], dci)
                    hess = T.set_subtensor(hess[self.nvar:(self.nvar + self.nineq), self.nvar:(self.nvar + self.nineq)],
                                           Sigma)  # T.triu(Sigma))
                    hess = T.set_subtensor(
                        hess[self.nvar:(self.nvar + self.nineq),
                             (self.nvar + self.nineq + self.neq):],
                        -T.eye(self.nineq)
                    )
                hess = T.triu(hess) + T.triu(hess).T
                hess = hess - T.diag(T.diagonal(hess) / 2.0)

        # construct expression for symmetric linear system solve
        lin_soln = sym_solve(self.M_dev, self.b_dev)

        # if using L-BFGS, get the expression for the descent direction
        if self.lbfgs:
            dz, dz_sqr = self.lbfgs_builder()

        # compile expressions into device functions
        if precompile:
            self.cost = f_func
        else:
            self.cost = theano.function(inputs=[self.x_dev], outputs=self.f)

        if precompile:
            self.barrier_cost_grad = barrier_cost_grad
        else:
            self.barrier_cost_grad = theano.function(inputs=[self.x_dev, self.s_dev],
                                                     outputs=barrier_cost_grad, on_unused_input='ignore')

        if precompile:
            self.grad = grad
        else:
            self.grad = theano.function(inputs=[self.x_dev, self.s_dev, self.lambda_dev],
                                        outputs=grad, on_unused_input='ignore')

        if self.lbfgs:
            self.lbfgs_dir_func = theano.function(inputs=[self.x_dev, self.s_dev, self.lambda_dev, self.g_dev,
                                                          self.zeta_dev, self.S_dev, self.Y_dev, self.SS_dev,
                                                          self.L_dev, self.D_dev, self.B_dev],
                                                  outputs=dz, on_unused_input='ignore')
            if dz_sqr is not None:
                self.lbfgs_dir_func_sqr = theano.function(inputs=[self.x_dev, self.s_dev, self.lambda_dev, self.g_dev,
                                                                  self.zeta_dev, self.s_dev, self.Y_dev, self.SS_dev,
                                                                  self.L_dev, self.D_dev, self.B_dev],
                                                          outputs=dz_sqr, on_unused_input='ignore')
        else:
            if precompile:
                self.hess = hess
            else:
                self.hess = theano.function(
                    inputs=[self.x_dev, self.s_dev, self.lambda_dev],
                    outputs=hess, on_unused_input='ignore'
                )

        if precompile:
            self.phi = phi
        else:
            self.phi = theano.function(
                inputs=[self.x_dev, self.s_dev],
                outputs=phi, on_unused_input='ignore'
            )

        if precompile:
            self.dphi = dphi
        else:
            self.dphi = theano.function(
                inputs=[self.x_dev, self.s_dev, self.dz_dev],
                outputs=dphi, on_unused_input='ignore'
            )

        self.eigh = theano.function(
            inputs=[self.M_dev],
            outputs=eigvalsh(self.M_dev, T.eye(self.M_dev.shape[0])),
        )

        self.sym_solve_cmp = theano.function(
            inputs=[self.M_dev, self.b_dev],
            outputs=lin_soln,
        )

        # self.gen_solve = theano.function(
        #    inputs=[self.M_dev, self.b_dev],
        #    outputs=gen_solve,
        # )

        if self.neq or self.nineq:
            if precompile:
                self.con = con
            else:
                self.con = theano.function(
                    inputs=[self.x_dev, self.s_dev],
                    outputs=con, on_unused_input='ignore'
                )

            if precompile:
                self.jaco = jaco
            else:
                self.jaco = theano.function(
                    inputs=[self.x_dev],
                    outputs=jaco,
                    on_unused_input='ignore',
                )

            if precompile:
                self.init_lambda = init_lambda
            else:
                self.init_lambda = theano.function(
                    inputs=[self.x_dev],
                    outputs=init_lambda,
                )

        if self.nineq:
            if precompile:
                self.init_slack = init_slack
            else:
                self.init_slack = theano.function(
                    inputs=[self.x_dev],
                    outputs=init_slack,
                )

        self.compiled = True

    def KKT(self, x, s, lda):
        """Calculate the first-order Karush-Kuhn-Tucker conditions. Irrelevant
           conditions are set to zero.
        """
        # kkt1 is the gradient of the Lagrangian with respect to x (weights)
        # kkt2 is the gradient of the Lagrangian with respect to s (slack variables)
        # kkt3 is the gradient of the Lagrangian with respect to lda[:self.neq] (equality constraints Lagrange
        #   multipliers)
        # kkt4 is the gradient of the Lagrangian with respect to lda[self.neq:] (inequality constraints Lagrange
        #   multipliers)
        kkts = self.grad(x, s, lda)

        if self.neq and self.nineq:
            kkt1 = kkts[:self.nvar]
            kkt2 = kkts[self.nvar:(self.nvar + self.nineq)] * s
            kkt3 = kkts[(self.nvar + self.nineq):(self.nvar + self.nineq + self.neq)]
            kkt4 = kkts[(self.nvar + self.nineq + self.neq):]
        elif self.neq:
            kkt1 = kkts[:self.nvar]
            kkt2 = self.float_dtype(0.0)
            kkt3 = kkts[(self.nvar + self.nineq):(self.nvar + self.nineq + self.neq)]
            kkt4 = self.float_dtype(0.0)
        elif self.nineq:
            kkt1 = kkts[:self.nvar]
            kkt2 = kkts[self.nvar:(self.nvar + self.nineq)] * s
            kkt3 = self.float_dtype(0.0)
            kkt4 = kkts[(self.nvar + self.nineq + self.neq):]
        else:
            kkt1 = kkts[:self.nvar]
            kkt2 = self.float_dtype(0.0)
            kkt3 = self.float_dtype(0.0)
            kkt4 = self.float_dtype(0.0)

        return kkt1, kkt2, kkt3, kkt4

    def lbfgs_init(self):
        """Initialize storage arrays for L-BFGS algorithm.
        """
        # initialize diagonal constant and storage arrays
        zeta = self.float_dtype(self.lbfgs_zeta)
        S = np.array([], dtype=self.float_dtype).reshape((self.nvar, 0))
        Y = np.array([], dtype=self.float_dtype).reshape((self.nvar, 0))
        SS = np.array([], dtype=self.float_dtype).reshape((0, 0))
        L = np.array([], dtype=self.float_dtype).reshape((0, 0))
        D = np.array([], dtype=self.float_dtype).reshape((0, 0))
        lbfgs_fail = 0

        return zeta, S, Y, SS, L, D, lbfgs_fail

    def lbfgs_builder(self):
        """Build the L-BFGS search direction calculation in Theano.
        """
        # gradient vector
        self.g_dev = T.vector('self.g_dev')

        # initial guess of Hessian
        self.zeta_dev = T.scalar('self.zeta_dev')

        # storage arrays
        self.S_dev = T.matrix('S_dev')
        self.Y_dev = T.matrix('Y_dev')
        self.SS_dev = T.matrix('SS_dev')
        self.L_dev = T.matrix('L_dev')
        self.D_dev = T.matrix('D_dev')

        # Jacobian transposed
        if self.neq or self.nineq:
            self.B_dev = T.matrix('B_dev')
        else:
            self.B_dev = T.scalar('B_dev')

        # get the current number of L-BFGS updates
        m_lbfgs = self.S_dev.shape[1]

        if self.neq or self.nineq:
            # For constrained problems, the search direction is
            #
            #     dz = -H^(-1)*g
            #
            # where g is the gradient and the approximate Hessian is
            #          _                         _     _         _   _             _        _                 _
            #         | zeta*I,   0,    dce,  dci |   | zeta*S, Y |*| zeta*S^T*S, L |^(-1)*| zeta*S^T, 0, 0, 0 |
            #     H = |    0,   Sigma,   0,   -I  | _ |    0,   0 | |_   L^T,    -D_|      |_  Y^T,    0, 0, 0_|
            #         |   -I,     0,  -eta*I,  0  |   |    0,   0 |
            #         |_ dce^T, dci^T,   0,    0 _|   |_   0,   0_|
            #          _                         _     _ _          _            _
            #         |       A.            B     |   | W |*M^(-1)*|_W^T, 0, 0, 0_|
            #       = |                           | _ | 0 |                         = Z - U*M^(-1)*U^T.
            #         |      B^T,           D     |   | 0 |
            #         |_                         _|   |_0_|
            #
            # The approximate Hessian is inverted using the Woodbury matrix identity:
            #
            #     H^(-1) = Z^(-1) - Z^(-1)*U*(U^T*Z^(-1)*U - M)^(-1)*U^T*Z^(-1).

            # construct diagonal of 'A'
            Adiag = self.zeta_dev * T.ones((self.nvar, 1))
            if self.nineq:
                Sigma = (
                    self.lambda_dev[self.neq:] / (self.s_dev + self.eps)).reshape((self.nineq, 1))
                Adiag = T.concatenate([Adiag, Sigma], axis=0)

            if self.nvar + self.nineq == self.neq + self.nineq:
                # Jacobian is square and full-rank

                # calculate -Z^(-1)*g using the inverse of a block matrix
                v01 = sym_solve(
                    self.B_dev,
                    self.g_dev[:self.nvar +
                               self.nineq].reshape((self.neq + self.nineq, 1)),
                )
                v02 = sym_solve(
                    self.B_dev.T,
                    self.g_dev[self.nvar +
                               self.nineq:].reshape((self.neq + self.nineq, 1)),
                )
                v03 = -Adiag * sym_solve(self.B_dev, v02)
                Zg = T.concatenate([v02, v01 + v03], axis=0)

                # construct U
                W = T.concatenate(
                    [self.zeta_dev * self.S_dev, self.Y_dev], axis=1)
                if self.nineq:
                    W = T.concatenate(
                        [W, T.zeros((self.nineq, 2 * m_lbfgs))], axis=0)

                # calculate -Z^(-1)*U*(U^T*Z^(-1)*U - M)^(-1)*U^T*Z^(-1)*g (skipped if m_lbgs=0)
                invB_W = sym_solve(self.B_dev, W)
                M0 = T.concatenate(
                    [self.zeta_dev * self.SS_dev, self.L_dev], axis=1)
                M1 = T.concatenate([self.L_dev.T, -self.D_dev], axis=1)
                Minv = T.concatenate([M0, M1], axis=0)
                v10 = T.dot(W.T, Zg[:self.nvar + self.nineq])
                v11 = -sym_solve(Minv, v10)
                X10 = T.concatenate(
                    [T.zeros((self.nvar + self.nineq, 2 * m_lbfgs)), invB_W], axis=0)
                XZg = T.dot(X10, v11)

                # combine -Z^(-1)*g and -Z^(-1)*U*(U^T*Z^(-1)*U - M)^(-1)*U^T*Z^(-1)*g
                dz_sqr = ifelse(T.gt(m_lbfgs, 0), Zg - XZg, Zg)

            # Jacobian is not square or is rank-deficient

            # calculate some basic matrices from Z^(-1)
            BT_invA = self.B_dev.T
            BT_invA = T.dot(BT_invA, diag(1.0 / Adiag.reshape((Adiag.size,))))
            BT_invA_B = T.dot(BT_invA, self.B_dev)

            if self.neq:
                # regularize if the equality constraints Jacobian is ill-conditioned
                w = eigh(BT_invA_B[:self.neq, :self.neq])[0]
                rcond = T.min(T.abs_(w)) / T.max(T.abs_(w))
                BT_invA_B = ifelse(
                    T.le(rcond, self.eps),
                    T.inc_subtensor(BT_invA_B[:self.neq, :self.neq],
                                    self.reg_coef * self.eta * (self.mu_dev ** self.beta) * T.eye(self.neq)), BT_invA_B)

            # calculate -Z^(-1)*g using the inverse of a block matrix
            v00 = T.dot(
                BT_invA, self.g_dev[:self.nvar + self.nineq].reshape((self.nvar + self.nineq, 1)))
            v01 = sym_solve(BT_invA_B, v00)
            v02 = (self.g_dev[:self.nvar + self.nineq].reshape((self.nvar + self.nineq, 1)) / Adiag -
                   T.dot(BT_invA.T, v01))
            v03 = - \
                sym_solve(
                    BT_invA_B, self.g_dev[self.nvar + self.nineq:].reshape((self.neq + self.nineq, 1)))
            v04 = -T.dot(BT_invA.T, v03)
            Zg = T.concatenate([v02 + v04, v01 + v03], axis=0)

            # construct U
            W = T.concatenate([self.zeta_dev * self.S_dev, self.Y_dev], axis=1)
            if self.nineq:
                W = T.concatenate(
                    [W, T.zeros((self.nineq, 2 * m_lbfgs))], axis=0)

            # calculate -Z^(-1)*U*(U^T*Z^(-1)*U - M)^(-1)*U^T*Z^(-1)*g (skipped if m_lbgs=0)
            BT_gmaW = T.dot(self.B_dev.T, W) / self.zeta_dev
            X00 = -sym_solve(BT_invA_B, BT_gmaW)
            X01 = W / self.zeta_dev + T.dot(BT_invA.T, X00)
            X02 = T.dot(W.T, X01)
            M0 = T.concatenate(
                [self.zeta_dev * self.SS_dev, self.L_dev], axis=1)
            M1 = T.concatenate([self.L_dev.T, -self.D_dev], axis=1)
            Minv = T.concatenate([M0, M1], axis=0)
            v10 = T.dot(W.T, Zg[:self.nvar + self.nineq])
            v11 = sym_solve(X02 - Minv, v10)
            X10 = T.concatenate([X01, -X00], axis=0)
            XZg = T.dot(X10, v11)

            # combine -Z^(-1)*g and -Z^(-1)*U*(U^T*Z^(-1)*U - M)^(-1)*U^T*Z^(-1)*g
            dz = ifelse(T.gt(m_lbfgs, 0), Zg - XZg, Zg)
        else:
            # For unconstrained problems, calculate the search direction
            #
            #     dz = -H*g
            #
            # where g is the gradient and the approximate inverse Hessian is
            #                  _        _   _                                      _   _       _
            #     H = gma*I + |_S, gma*Y_|*| R^(-T)*(D + gma*Y^T*Y)*R^(-1), -R^(-T) |*|   S^T   |
            #                              |_             -R^(-1),               0 _| |_gma*Y^T_|
            #
            #       = gma*I + W*Q*W.T.
            #
            # The substitutions gma=zeta, R=L, and Y^T*Y=S^T*S are made to save variables.

            # calculate -gma*I*g
            Hg = self.zeta_dev * self.g_dev.reshape((self.nvar, 1))

            # calculate -W*Q*W.T*g (skipped if m_lbfgs=0)
            W = T.concatenate([self.S_dev, self.zeta_dev * self.Y_dev], axis=1)
            WT_g = T.dot(W.T, self.g_dev)
            B = -sym_solve(self.L_dev, WT_g[:m_lbfgs].reshape((m_lbfgs, 1)))
            A = (-sym_solve(self.L_dev.T, T.dot(self.D_dev + self.zeta_dev * self.SS_dev, B)) -
                 sym_solve(self.L_dev.T, WT_g[m_lbfgs:].reshape((m_lbfgs, 1))))
            Hg_update = T.dot(W, T.concatenate([A, B], axis=0))

            # combine -gma*I*g and -W*Q*W.T*g
            dz = ifelse(T.gt(m_lbfgs, 0), Hg + Hg_update, Hg)

        if self.nvar + self.nineq == self.neq + self.nineq:
            # if constraints Jacobian is square, return both rank-deficient and full-rank search direction expressions
            return dz, dz_sqr
        else:
            # if constraints Jacobian is rectangular or the problem is unconstrained, return only one expression
            return dz, None

    def lbfgs_dir(self, x, s, lda, g, zeta, S, Y, SS, L, D):
        """Calculate the search direction for the L-BFGS algorithm.
        """
        # calculate the step direction
        if self.neq or self.nineq:
            reduce = False
            B = self.jaco(x)
            if B.shape[0] == B.shape[1]:
                if np.linalg.cond(B) < 1.0 / self.eps:
                    # B is invertible, reduce problem
                    reduce = True
            if reduce:
                # if constraints Jacobian is square and full-rank
                dz = self.lbfgs_dir_func_sqr(
                    x, s, lda, g, zeta, S, Y, SS, L, D, B)
            else:
                # if constraints Jacobian is rectangular or rank-deficient
                dz = self.lbfgs_dir_func(x, s, lda, g, zeta, S, Y, SS, L, D, B)

                # inefficient prototype for testing
                # m_lbfgs = S.shape[1]
                # H = np.zeros((self.nvar+2*self.nineq+self.neq, self.nvar+2*self.nineq+self.neq))
                # H[:self.nvar,:self.nvar] = zeta*np.eye(self.nvar)
                # if self.nineq:
                #    Sigma = (lda[self.neq:]/(s+self.eps)).reshape((self.nineq,))
                #    H[self.nvar:self.nvar+self.nineq, self.nvar:self.nvar+self.nineq] = \
                #        np.diag(lda[self.neq:]/(s+self.eps))
                # H[self.nvar+self.nineq:, :self.nvar+self.nineq] = B.T
                # H[:self.nvar+self.nineq, self.nvar+self.nineq:] = B
                #
                # Zg_new = self.sym_solve_cmp(H, g.reshape((g.size,1)))
                #
                # if m_lbfgs > 0:
                #    M0 = np.concatenate([zeta*SS, L], axis=1)
                #    M1 = np.concatenate([L.T, -D], axis=1)
                #    Minv = np.concatenate([M0, M1], axis=0)
                #    W = np.concatenate([zeta*S, Y], axis=1)
                #    H[:self.nvar, :self.nvar] = (H[:self.nvar, :self.nvar] -
                #                                 np.dot(W, self.sym_solve_cmp(Minv,
                #                                                              W.T.reshape((W.size/self.nvar, self.nvar)))))
                #
                # dz = self.sym_solve_cmp(H, g.reshape((g.size,1)))
        else:
            # if problem is unconstrained
            dz = self.lbfgs_dir_func(x, s, lda, g, zeta, S, Y, SS, L, D, None)

            # inefficient prototype for testing
            # m_lbfgs = S.shape[1]
            # H = 1.0/zeta*np.eye(self.nvar)
            #
            # SStrue = np.dot(S.T, S)
            # Ltrue = np.tril(np.dot(S.T, Y), -1)
            #
            # if S.size > 0:
            #    M0 = np.concatenate([1.0/zeta*SStrue, Ltrue], axis=1)
            #    M1 = np.concatenate([Ltrue.T, -D], axis=1)
            #    Minv = np.concatenate([M0, M1], axis=0)
            #    W = np.concatenate([1.0/zeta*S, Y], axis=1)
            #    H -= np.dot(W, self.sym_solve_cmp(Minv, W.T.reshape((W.size/self.nvar, self.nvar))))
            #
            # dz = self.sym_solve_cmp(H, g.reshape((g.size,1)))

        return dz.reshape((dz.size,))

    # def lbfgs_curv_perturb(self, dx, dg):
    #     """Perturb the curvature of the L-BFGS update when
    #        np.dot(dg, dx) <= 0.0 to maintain positive definiteness
    #        of the Hessian approximation.
    #     """

    #     if np.dot(dg, dx) <= 0.0:
    #         # L-BFGS update is not positive definite, cut most negative value of the gradient displacement until it is
    #         # close to zero or the update becomes positive semidefinite
    #         idx = np.argmin(dg * dx)
    #         while np.dot(dg, dx) < -np.sqrt(self.eps) and dg[idx] * dx[idx] < -np.sqrt(self.eps):
    #             dg[idx] *= 0.5
    #     if np.dot(dg, dx) < np.sqrt(self.eps) and (self.neq or self.nineq):
    #         # if the above procedure did not work, perturb the negative gradient displacements until the L-BFGS update
    #         # is positive definite
    #         dc_new = self.jaco(x_new)
    #         dc_old = self.jaco(x_old)
    #         dcc = np.dot(dc_old, g_old[self.nvar + self.nineq:]) - \
    #             np.dot(dc_new, g_new[self.nvar + self.nineq:])
    #         self.delta = self.delta0
    #         dg_new = np.copy(dg)
    #         inp = np.dot(dg_new, dx)
    #         while inp < np.sqrt(self.eps) and np.linalg.norm(dg_new) > np.sqrt(self.eps) and not np.isinf(inp):
    #             dg_new = np.copy(dg)
    #             mask = np.where(dg_new * dx < np.sqrt(self.eps))
    #             dg_new[mask] = dg[mask] + self.delta * \
    #                 np.sign(dx[mask]) * np.abs(dcc[mask])
    #             self.delta *= 2.0
    #             inp = np.dot(dg_new, dx)
    #         dg = np.copy(dg_new)

        # return the perturbed gradient displacement
        return dg

    def lbfgs_update(self, x_old, x_new, g_old, g_new, zeta, S, Y, SS, L, D, lbfgs_fail):
        """Update stored arrays for the next L-BFGS iteration
        """
        # calculate displacements with respect to the weights and gradient
        dx = x_new - x_old
        dg = g_old[:self.nvar] - g_new[:self.nvar]

        # curvature perturbation (not used)
        # dg = self.lbfgs_curv_perturb(dx, dg)

        # calculate updated zeta
        if self.neq or self.nineq:
            zeta_new = np.dot(dg, dx) / (np.dot(dx, dx) + self.eps)
        else:
            zeta_new = np.dot(dg, dx) / (np.dot(dg, dg) + self.eps)
        if np.dot(dx, dg) > np.sqrt(self.eps) and zeta_new > np.sqrt(self.eps):
            # if initial Hessian approximation is positive definite, update storage arrays (see [2] for definitions)
            zeta = zeta_new
            if S.shape[1] > self.lbfgs:
                # if S and Y exceed memory limit, remove oldest displacements
                S[:, :-1] = S[:, 1:]
                Y[:, :-1] = Y[:, 1:]
                SS[:-1, :-1] = SS[1:, 1:]
                L[:-1, :-1] = L[1:, 1:]
                D[:-1, :-1] = D[1:, 1:]
            else:
                # otherwise, expand arrays
                lsize = S.shape[1] + 1
                S = np.concatenate(
                    [S, np.zeros((self.nvar, 1), dtype=self.float_dtype)], axis=1)
                Y = np.concatenate(
                    [Y, np.zeros((self.nvar, 1), dtype=self.float_dtype)], axis=1)
                SS = np.concatenate(
                    [SS, np.zeros((1, lsize - 1), dtype=self.float_dtype)], axis=0)
                SS = np.concatenate(
                    [SS, np.zeros((lsize, 1), dtype=self.float_dtype)], axis=1)
                L = np.concatenate(
                    [L, np.zeros((1, lsize - 1), dtype=self.float_dtype)], axis=0)
                L = np.concatenate(
                    [L, np.zeros((lsize, 1), dtype=self.float_dtype)], axis=1)
                D = np.concatenate(
                    [D, np.zeros((1, lsize - 1), dtype=self.float_dtype)], axis=0)
                D = np.concatenate(
                    [D, np.zeros((lsize, 1), dtype=self.float_dtype)], axis=1)

            S[:, -1] = dx
            Y[:, -1] = dg

            if self.neq or self.nineq:
                SS_update = np.dot(S.T, dx.reshape((self.nvar, 1)))
            else:
                # this is YY_update
                SS_update = np.dot(Y.T, dg.reshape((self.nvar, 1)))

            # update storage arrays (this is YY for unconstrained)
            SS[:, -1] = SS_update.reshape((SS_update.size,))
            SS[-1, :] = SS_update.reshape((SS_update.size,))
            lsize = SS.shape[1]
            SS = SS.reshape((lsize, lsize))

            if self.neq or self.nineq:
                L_update = np.dot(dx.reshape((1, self.nvar)), Y)
                L[-1, :] = L_update
                L[-1, -1] = self.float_dtype(0.0)
            else:
                # this is R_update and R
                L_update = np.dot(S.T, dg.reshape(
                    (self.nvar, 1))).reshape((S.shape[1],))
                L[:, -1] = L_update
            L = L.reshape((lsize, lsize))

            D_update = np.dot(dx, dg)
            D[-1, -1] = D_update
            D = D.reshape((lsize, lsize))

            # reset L-BFGS failure counter
            lbfgs_fail = 0
        else:
            # increment L-BFGS failure counter
            lbfgs_fail += 1

        if lbfgs_fail > self.lbfgs_fail_max and S.shape[1] > 0:
            # if the L-BFGS fairlure counter exceeds the maximum number of failures, reset initial Hessian approximation
            # and storage arrays
            if self.verbosity > 2:
                print('Max failures reached, resetting storage arrays.')
            zeta, S, Y, SS, L, D, lbfgs_fail = self.lbfgs_init()

        # return initial Hessian approximation, storage arrays, and L-BFGS failure counter
        return zeta, S, Y, SS, L, D, lbfgs_fail

    def reghess(self, Hc):
        """Regularize the Hessian to avoid ill-conditioning and to escape saddle
           points.
        """
        # compute eigenvalues and condition number
        w = self.eigh(Hc)
        rcond = np.min(np.abs(w)) / np.max(np.abs(w))

        if rcond <= self.eps or (self.neq + self.nineq) != np.sum(w < -self.eps):
            # if the Hessian is ill-conditioned or the matrix inertia is undesireable, regularize the Hessian
            if rcond <= self.eps and self.neq:
                # if the Hessian is ill-conditioned, regularize by replacing some zeros with a small magnitude diagonal
                # matrix
                ind1 = self.nvar + self.nineq
                ind2 = ind1 + self.neq
                Hc[ind1:ind2, ind1:ind2] -= self.reg_coef * self.eta * \
                    (self.mu_host ** self.beta) * np.eye(self.neq)
            if self.delta == 0.0:
                # if the diagonal shift coefficient is zero, set to initial value
                self.delta = self.delta0
            else:
                # prevent the diagonal shift coefficient from becoming too small
                self.delta = np.max([self.delta / 2, self.delta0])
            # regularize Hessian with diagonal shift matrix (delta*I) until matrix inertia condition is satisfied
            Hc[:self.nvar, :self.nvar] += self.delta * np.eye(self.nvar)
            w = self.eigh(Hc)
            while (self.neq + self.nineq) != np.sum(w < -self.eps):
                Hc[:self.nvar, :self.nvar] -= self.delta * np.eye(self.nvar)
                self.delta *= 10.0
                Hc[:self.nvar, :self.nvar] += self.delta * np.eye(self.nvar)
                w = self.eigh(Hc)

        # return regularized Hessian
        return Hc

    def step(self, x, dx):
        """Golden section search used to determine the maximum
           step length for slack variables and Lagrange multipliers
           using the fraction-to-the-boundary rule.
        """
        GOLD = (np.sqrt(5.0) + 1.0) / 2.0

        a = 0.0
        b = 1.0
        if np.all(x + b * dx >= (1.0 - self.tau) * x):
            return b
        else:
            c = b - (b - a) / GOLD
            d = a + (b - a) / GOLD
            while np.abs(b - a) > GOLD * self.Xtol:
                if np.any(x + d * dx < (1.0 - self.tau) * x):
                    b = np.copy(d)
                else:
                    a = np.copy(d)
                if c > a:
                    if np.any(x + c * dx < (1.0 - self.tau) * x):
                        b = np.copy(c)
                    else:
                        a = np.copy(c)

                c = b - (b - a) / GOLD
                d = a + (b - a) / GOLD

            return a

    def search(self, x0, s0, lda0, dz, alpha_smax, alpha_lmax):
        """Backtracking line search to find a solution that leads
           to a smaller value of the Lagrangian within the confines
           of the maximum step length for the slack variables and
           Lagrange multipliers found using class function 'step'.
        """
        # extract search directions along x, s, and lda (weights, slacks, and multipliers)
        dx = dz[:self.nvar]
        if self.nineq:
            ds = dz[self.nvar:(self.nvar + self.nineq)]

        if self.neq or self.nineq:
            dl = dz[(self.nvar + self.nineq):]
        else:
            dl = self.float_dtype(0.0)
            alpha_lmax = self.float_dtype(0.0)

        x = np.copy(x0)
        s = np.copy(s0)
        phi0 = self.phi(x0, s0)
        dphi0 = self.dphi(x0, s0, dz[:self.nvar + self.nineq])
        correction = False
        if self.nineq:
            # step search when there are inequality constraints
            if self.phi(x0 + alpha_smax * dx, s0 + alpha_smax * ds) > phi0 + alpha_smax * self.eta * dphi0:
                # second-order correction
                c_old = self.con(x0, s0)
                c_new = self.con(x0 + alpha_smax * dx, s0 + alpha_smax * ds)
                if np.sum(np.abs(c_new)) > np.sum(np.abs(c_old)):
                    # infeasibility has increased, attempt to correct
                    A = self.jaco(x0).T
                    try:
                        # calculate a feasibility restoration direction
                        dz_p = -self.sym_solve_cmp(
                            A,
                            c_new.reshape((self.nvar + self.nineq, 1))
                        ).reshape((self.nvar + self.nineq,))
                    except:
                        # if the Jacobian is not invertible, find the minimum norm solution instead
                        dz_p = -np.linalg.lstsq(A, c_new, rcond=None)[0]
                    if (self.phi(x0 + alpha_smax * dx + dz_p[:self.nvar], s0 + alpha_smax * ds + dz_p[self.nvar:]) <=
                            phi0 + alpha_smax * self.eta * dphi0):
                        alpha_corr = self.step(
                            s0, alpha_smax * ds + dz_p[self.nvar:])
                        if (self.phi(x0 + alpha_corr * (alpha_smax * dx + dz_p[:self.nvar]),
                                     s0 + alpha_corr * (alpha_smax * ds + dz_p[self.nvar:])) <=
                                phi0 + alpha_smax * self.eta * dphi0):
                            if self.verbosity > 2:
                                print(
                                    'Second-order feasibility correction accepted')
                            # correction accepted
                            correction = True
                if not correction:
                    # infeasibility has not increased, no correction necessary
                    alpha_smax *= self.tau
                    alpha_lmax *= self.tau
                    while self.phi(x0 + alpha_smax * dx, s0 + alpha_smax * ds) > phi0 + alpha_smax * self.eta * dphi0:
                        # backtracking line search
                        if (np.sqrt(np.linalg.norm(alpha_smax * dx) ** 2 + np.linalg.norm(alpha_lmax * ds) ** 2) <
                                self.eps):
                            # search direction is unreliable to machine precision, stop solver
                            if self.verbosity > 2:
                                print(
                                    'Search direction is unreliable to machine precision.')
                            self.signal = -2
                            return x0, s0, lda0
                        alpha_smax *= self.tau
                        alpha_lmax *= self.tau
            # update slack variables
            if correction:
                s = s0 + alpha_corr * (alpha_smax * ds + dz_p[self.nvar:])
            else:
                s = s0 + alpha_smax * ds
        else:
            # step search for only equality constraints or unconstrained problems
            if self.phi(x0 + alpha_smax * dx, s0) > phi0 + alpha_smax * self.eta * dphi0:
                if self.neq:
                    # second-order correction
                    c_old = self.con(x0, s0)
                    c_new = self.con(x0 + alpha_smax * dx, s0)
                    if np.sum(np.abs(c_new)) > np.sum(np.abs(c_old)):
                        # infeasibility has increased, attempt to correct
                        A = self.jaco(x0).T
                        try:
                            # calculate a feasibility restoration direction
                            dz_p = -self.sym_solve_cmp(
                                A,
                                c_new.reshape((self.nvar, self.nineq, 1))
                            ).reshape((self.nvar + self.nineq,))
                        except:
                            # if the Jacobian is not invertible, find the minimum norm solution instead
                            dz_p = -np.linalg.lstsq(A, c_new, rcond=None)[0]
                        if self.phi(x0 + alpha_smax * dx + dz_p, s0) <= phi0 + alpha_smax * self.eta * dphi0:
                            # correction accepted
                            if self.verbosity > 2:
                                print(
                                    'Second-order feasibility correction accepted')
                            alpha_corr = self.float_dtype(1.0)
                            correction = True
                if not correction:
                    # infeasibility has not increased, no correction necessary
                    alpha_smax *= self.tau
                    alpha_lmax *= self.tau
                    while self.phi(x0 + alpha_smax * dx, s0) > phi0 + alpha_smax * self.eta * dphi0:
                        # backtracking line search
                        if np.linalg.norm(alpha_smax * dx) < self.eps:
                            # search direction is unreliable to machine precision, stop solver
                            if self.verbosity > 2:
                                print(
                                    'Search direction is unreliable to machine precision.')
                            self.signal = -2
                            return x0, s0, lda0
                        alpha_smax *= self.tau
                        alpha_lmax *= self.tau
        # update weights
        if correction:
            x = x0 + alpha_corr * (alpha_smax * dx + dz_p[:self.nvar])
        else:
            x = x0 + alpha_smax * dx

        # update multipliers (if applicable)
        if self.neq or self.nineq:
            lda = lda0 + alpha_lmax * dl
        else:
            lda = np.copy(lda0)

        # return updated weights, slacks, and multipliers
        return x, s, lda

    def solve(self, x0=None, s0=None, lda0=None, force_recompile=False):
        """Main solver function that initiates, controls the iteraions, and
         performs the primary operations of the line search primal-dual
         interior-point method.
        """
        # check if weights, slacks, or multipliers are given as input
        if x0 is not None:
            self.x0 = x0
        if s0 is not None:
            self.s0 = s0
        if lda0 is not None:
            self.lda0 = lda0

        # weights must be initialized and have length greater than zero
        assert (self.x0 is not None) and (self.x0.size > 0)
        # weights should be a one-dimensional array
        assert self.x0.size == self.x0.shape[0]
        # set the variable counter equal to the number of weights
        self.nvar = self.x0.size
        # cast weights to float_dtype
        self.x0 = self.float_dtype(self.x0)

        # validate class members
        self.validate()

        # if expressions are not compiled or force_recompile=True, compile expressions into functions
        if not self.compiled or force_recompile:
            self.compile()

        # intialize weighs, slacks, and multipliers
        x = self.x0
        if self.nineq:
            if self.s0 is None:
                s = self.init_slack(x)
            else:
                s = self.s0.astype(self.float_dtype)
            self.mu_host = self.mu
        else:
            s = np.array([], dtype=self.float_dtype)
            self.mu_host = self.Ktol
            self.mu_dev.set_value(self.float_dtype(self.mu_host))

        if self.neq or self.nineq:
            self.nu_host = self.nu
            self.nu_dev.set_value(self.float_dtype(self.nu_host))
            if self.lda0 is None:
                lda = self.init_lambda(x)
                if self.nineq and self.neq:
                    lda_ineq = lda[self.neq:]
                    lda_ineq[lda_ineq < self.float_dtype(
                        0.0)] = self.float_dtype(self.Ktol)
                    lda[self.neq:] = lda_ineq
                elif self.nineq:
                    lda[lda < self.float_dtype(
                        0.0)] = self.float_dtype(self.Ktol)
            else:
                lda = self.lda0.astype(self.float_dtype)
        else:
            lda = np.array([], dtype=self.float_dtype)

        # initialize diagonal shift coefficient
        self.delta = self.float_dtype(0.0)

        # calculate the initial KKT conditions
        kkt = self.KKT(x, s, lda)

        if self.lbfgs:
            # if using L-BFGS algorithm, initialize Hessian approximation, storage arrays, prior weights, and gradient
            zeta, S, Y, SS, L, D, lbfgs_fail = self.lbfgs_init()
            x_old = np.copy(x)
            g = -self.grad(x, s, lda)

        if self.verbosity > 0:
            if self.lbfgs:
                print(
                    'Searching for a feasible local minimizer using L-BFGS to approximate the Hessian.')
            else:
                print(
                    'Searching for a feasible local minimizer using the exact Hessian.')

        iter_count = 0
        if self.Ftol is not None:
            # if Ftol is set, calculate the prior cost
            f_past = self.cost(x)

        Ktol_converged = False
        Ftol_converged = False

        # initialize optimization return signal
        self.signal = 0

        for outer in range(self.niter):
            # begin outer iterations where the barrier parameter is adjusted

            # determine if the current point has converged to Ktol precision using KKT conditions; if True, solution
            # found
            if all([np.linalg.norm(kkt[0]) <= self.Ktol, np.linalg.norm(kkt[1]) <= self.Ktol,
                    np.linalg.norm(kkt[2]) <= self.Ktol, np.linalg.norm(kkt[3]) <= self.Ktol]):
                self.signal = 1
                Ktol_converged = True
                break

            if self.verbosity > 0 and self.nineq:
                print('OUTER ITERATION {}'.format(outer + 1))

            for inner in range(self.miter):
                # begin inner iterations where the barrier parameter is held fixed

                # check convergence to muTol precision using the KKT conditions; if True, break from the inner loop
                muTol = np.max([self.Ktol, self.mu_host])
                if all([np.linalg.norm(kkt[0]) <= muTol, np.linalg.norm(kkt[1]) <= muTol,
                        np.linalg.norm(kkt[2]) <= muTol, np.linalg.norm(kkt[3]) <= muTol]):
                    if not self.neq and not self.nineq:
                        self.signal = 1
                        Ktol_converged = True
                    break

                if self.verbosity > 0:
                    msg = []
                    if self.nineq:
                        msg.append('* INNER ITERATION {}'.format(inner + 1))
                    else:
                        msg.append('ITERATION {}'.format(iter_count + 1))
                    if self.verbosity > 1:
                        msg.append('f(x) = {}'.format(self.cost(x)))
                    if self.verbosity > 2:
                        msg.append(
                            '|dL/dx| = {}'.format(np.linalg.norm(kkt[0])))
                        msg.append(
                            '|dL/ds| = {}'.format(np.linalg.norm(kkt[1])))
                        msg.append('|ce| = {}'.format(np.linalg.norm(kkt[2])))
                        msg.append(
                            '|ci-s| = {}'.format(np.linalg.norm(kkt[3])))
                    print(', '.join(msg))

                if self.lbfgs:
                    # if using the L-BFGS algorithm, calculate the new and prior gradients and weights and update
                    # storage arrays
                    if inner > 0 or outer > 0:
                        g_old = -self.grad(x_old, s, lda)
                        g_new = -self.grad(x, s, lda)
                        zeta, S, Y, SS, L, D, lbfgs_fail = self.lbfgs_update(x_old, x, g_old, g_new, zeta, S, Y, SS, L,
                                                                             D, lbfgs_fail)
                        x_old = np.copy(x)
                        g = np.copy(g_new)
                    # calculate the search direction
                    dz = self.lbfgs_dir(x, s, lda, g, zeta, S, Y, SS, L, D)
                else:
                    # calculate the gradient and Hessian and regularize the Hessian if necessary to maintain appropriate
                    # matrix inertia
                    g = -self.grad(x, s, lda)
                    Hc = self.reghess(self.hess(x, s, lda))
                    # calculate the search direction
                    dz = self.sym_solve_cmp(Hc, g.reshape(
                        (g.size, 1))).reshape((g.size,))

                if self.neq or self.nineq:
                    # change sign definition for the multipliers' search direction
                    dz[self.nvar + self.nineq:] = -dz[self.nvar + self.nineq:]

                if self.neq or self.nineq:
                    # update the merit function parameter, if necessary
                    nu_thres = np.dot(
                        self.barrier_cost_grad(x, s),
                        dz[:self.nvar + self.nineq]
                    ) / (1 - self.rho) / np.sum(np.abs(self.con(x, s)))
                    if self.nu_host < nu_thres:
                        self.nu_host = self.float_dtype(nu_thres)
                        self.nu_dev.set_value(self.nu_host)

                if self.nineq:
                    # use fraction-to-the-boundary rule to make sure slacks and multipliers do not decrease too quickly
                    alpha_smax = self.step(
                        s, dz[self.nvar:(self.nvar + self.nineq)])
                    alpha_lmax = self.step(
                        lda[self.neq:], dz[(self.nvar + self.nineq + self.neq):])
                    # use a backtracking line search to update weights, slacks, and multipliers
                    x, s, lda = self.search(x, s, lda, dz, self.float_dtype(
                        alpha_smax), self.float_dtype(alpha_lmax))
                else:
                    # use a backtracking line search to update weights, slacks, and multipliers
                    x, s, lda = self.search(
                        x, s, lda, dz, self.float_dtype(1.0), self.float_dtype(1.0))

                iter_count += 1

                # calculate the updated KKT conditions
                kkt = self.KKT(x, s, lda)

                if all([self.Ftol is not None, not self.nineq, self.signal != -2]):
                    # for unconstrained and equality constraints only, calculate new cost and check Ftol convergence
                    f_new = self.cost(x)
                    if np.abs(f_past - f_new) <= np.abs(self.Ftol):
                        # converged to Ftol precision
                        self.signal = 2
                        Ftol_converged = True
                        break
                    else:
                        # did not converge, update past cost
                        f_past = f_new

                if self.signal == -2:
                    # a bad search direction was chosen, terminating
                    break

                if inner >= self.miter - 1:
                    if self.verbosity > 0 and self.nineq:
                        print('MAXIMUM INNER ITERATIONS EXCEEDED')

            if all([self.Ftol is not None, self.nineq, self.signal != -2]):
                # when problem has inequality constraints, calculate new cost and check Ftol convergence
                f_new = self.cost(x)
                if np.abs(f_past - f_new) <= np.abs(self.Ftol):
                    # converged to Ftol precision
                    self.signal = 2
                    Ftol_converged = True
                else:
                    # did not converge, update past cost
                    f_past = f_new

            if self.Ftol is not None and Ftol_converged:
                # if Ftol convergence reached, break because solution has been found
                break

            if self.signal == -2:
                # a bad search direction was chosen, terminating
                break

            if outer >= self.niter - 1:
                self.signal = -1
                if self.verbosity > 0:
                    if self.nineq:
                        print('MAXIMUM OUTER ITERATIONS EXCEEDED')
                    else:
                        print('MAXIMUM ITERATIONS EXCEEDED')
                break

            if self.nineq:
                # update the barrier parameter
                xi = self.nineq * \
                    np.min(s * lda[self.neq:]) / \
                    (np.dot(s, lda[self.neq:]) + self.eps)
                self.mu_host = (0.1 * np.min([0.05 * (1.0 - xi) / (xi + self.eps), 2.0]) ** 3 *
                                np.dot(s, lda[self.neq:]) / self.nineq)
                if self.float_dtype(self.mu_host) < self.float_dtype(0.0):
                    self.mu_host = 0.0
                self.mu_host = self.float_dtype(self.mu_host)
                self.mu_dev.set_value(self.mu_host)

        # assign class member variables to the solutions
        self.x = x
        self.s = s
        self.lda = lda
        self.kkt = kkt
        self.fval = self.cost(x)

        if self.verbosity >= 0:
            msg = []
            if self.signal == -2:
                msg.append(
                    'Terminated due to bad direction in backtracking line search')
            elif all([np.linalg.norm(kkt[0]) <= self.Ktol, np.linalg.norm(kkt[1]) <= self.Ktol,
                      np.linalg.norm(kkt[2]) <= self.Ktol, np.linalg.norm(kkt[3]) <= self.Ktol]):
                msg.append('Converged to Ktol tolerance')
            elif self.Ftol is not None and Ftol_converged:
                msg.append('Converged to Ftol tolerance')
            else:
                msg.append('Maximum iterations reached')
                outer = self.niter
                inner = 0

            if self.nineq:
                if outer > 1:
                    msg.append('after {} outer'.format(outer - 1))
                    msg.append('iterations' if outer > 2 else 'iteration')
                    msg.append('and')
                else:
                    msg.append('after')
                msg.append('{} inner'.format(inner))
                msg.append('iterations' if inner > 1 else 'iteration')
                msg.append('({} total).'.format(iter_count))
            else:
                msg.append('after {}'.format(iter_count))
                msg.append('iterations.' if iter_count > 1 else 'iteration.')
            print(' '.join(msg))
            if self.verbosity > 1:
                msg = []
                msg.append('FINAL: f(x) = {}'.format(self.cost(x)))
                if self.verbosity > 2:
                    msg.append('|dL/dx| = {}'.format(np.linalg.norm(kkt[0])))
                    msg.append('|dL/ds| = {}'.format(np.linalg.norm(kkt[1])))
                    msg.append('|ce| = {}'.format(np.linalg.norm(kkt[2])))
                    msg.append('|ci-s| = {}'.format(np.linalg.norm(kkt[3])))
                print(', '.join(msg))

        # return solution weights, slacks, multipliers, cost, and KKT conditions
        return self.x, self.s, self.lda, self.fval, self.kkt


def main():
    import sys
    import os

    def list2str(lst):
        return [str(l) for l in lst]

    # This main function provides example problems that may be used to help users write their code
    # (and to make sure modifications to IPM do not break the code). To call these test problems,
    # use command line arguments; e.g.
    #
    #     python pyipm.py 5
    #
    # to run the 5th problem. There are 10 problems total called by arguments 1-10.

    # To use L-BFGS to approximate the Hessian, set lbfgs to a positive integer to define the
    # number of iterations to store to make the Hessian approximation. Otherwise, set lbfgs to
    # False or 0.
    lbfgs = False

    # The verbosity level between from -1 up to 3 determines the amount of feedback the algorithm
    # gives to the user during the optimization.
    verbosity = 1

    # Setting Ftol (the function tolerance) can be a helpful secondary criteria for convergence;
    # by default, Ftol is unset and only Ktol is used (the KKT conditions tolerance).
    # E.g. on occasion, L-BFGS may converge slowly on the Rosenbrock example so Ftol=1.0E-8 can
    # be used as a safeguard.
    Ftol = 1.0E-8

    # x_dev is a device vector that must be predefined by the user and is used to build theano
    # expressions.
    x_dev = T.vector('x_dev')

    # get the problem number from the command line argument list.
    prob = int(sys.argv[1])

    # determine the floating-point type from the 'THEANO_FLAGS' environment variable.
    float_dtype = os.environ.get('THEANO_FLAGS')
    if float_dtype is not None:
        try:
            float_dtype = float_dtype.split('floatX=')[1]
        except IndexError:
            raise Exception(
                'Error: attribute "floatX" not defined in "THEANO_FLAGS" environment variable.')
        float_dtype = float_dtype.split(',')[0]
    else:
        raise Exception('Error: "THEANO_FLAGS" environment variable is unset.')
    if float_dtype.strip() == 'float32':
        float_dtype = np.float32
    else:
        float_dtype = np.float64

    # example problem definitions
    if prob == 1:
        print('minimize f(x, y) = x**2 - 4*x + y**2 - y - x*y')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)

        f = x_dev[0] ** 2 - 4 * x_dev[0] + \
            x_dev[1] ** 2 - x_dev[1] - x_dev[0] * x_dev[1]

        p = IPM(x0=x0, x_dev=x_dev, f=f, Ftol=Ftol, lbfgs=lbfgs,
                float_dtype=float_dtype, verbosity=verbosity)
        x, s, lda, fval, kkt = p.solve()

        gt = [float_dtype(3.0), float_dtype(2.0)]
        print('')
        print('Ground truth: [x, y] = [{}, {}]'.format(*gt))
        print('Solver solution: [x, y] = [{}, {}]'.format(*x))
        print('f(x, y) = {}'.format(fval))
    elif prob == 2:
        print('Find the global minimum of the 2D Rosenbrock function.')
        print('minimize f(x, y) = 100*(y - x**2)**2 + (1 - x)**2')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)

        f = 100 * (x_dev[1] - x_dev[0] ** 2) ** 2 + (1 - x_dev[0]) ** 2

        p = IPM(x0=x0, x_dev=x_dev, f=f, Ftol=Ftol, lbfgs=lbfgs,
                float_dtype=float_dtype, verbosity=verbosity)
        x, s, lda, fval, kkt = p.solve()

        gt = [float_dtype(1.0), float_dtype(1.0)]
        print('')
        print('Ground truth: [x, y] = [{}, {}]'.format(*gt))
        print('Solver solution: [{}, {}]'.format(*x))
        print('f(x, y) = {}'.format(fval))
    elif prob == 3:
        print('maximize f(x, y) = x + y subject to x**2 + y**2 = 1')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)

        f = -T.sum(x_dev)
        ce = T.sum(x_dev ** 2) - 1.0

        p = IPM(x0=x0, x_dev=x_dev, f=f, ce=ce, Ftol=Ftol, lbfgs=lbfgs,
                float_dtype=float_dtype, verbosity=verbosity)
        x, s, lda, fval, kkt = p.solve()

        gt = [float_dtype(np.sqrt(2.0) / 2.0), float_dtype(np.sqrt(2.0) / 2.0)]
        print('')
        print('Ground truth: [x, y] = [{}, {}]'.format(*gt))
        print('Solver solution: [x, y] = [{}, {}]'.format(*x))
        print('Lagrange multiplier: lda = {}'.format(lda))
        print('f(x, y) = {}'.format(-fval))
    elif prob == 4:
        print('maximize f(x, y) = (x**2)*y subject to x**2 + y**2 = 3')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)

        f = -(x_dev[0] ** 2) * x_dev[1]
        ce = T.sum(x_dev ** 2) - 3.0

        p = IPM(x0=x0, x_dev=x_dev, f=f, ce=ce, Ftol=Ftol, lbfgs=lbfgs,
                float_dtype=float_dtype, verbosity=verbosity)
        x, s, lda, fval, kkt = p.solve()

        gt = [
            float_dtype(np.sqrt(2.0)), float_dtype(1.0),
            -float_dtype(np.sqrt(2.0)), float_dtype(1.0),
            float_dtype(0.0), -float_dtype(-np.sqrt(3.0)),
        ]
        print('')
        print(
            'Ground truth: global max. @ [x, y] = [{}, {}] or [{}, {}], local max. @ [{}, {}]'.format(*gt))
        print('Solver solution: [x, y] = [{}, {}]'.format(*x))
        print('Lagrange multipler: lda = {}'.format(lda))
        print('f(x, y) = {}'.format(-fval))
    elif prob == 5:
        print('minimize f(x, y) = x**2 + 2*y**2 + 2*x + 8*y subject to -x - 2*y + 10 <= 0, x >= 0, y >= 0')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)

        f = x_dev[0] ** 2 + 2.0 * x_dev[1] ** 2 + \
            2.0 * x_dev[0] + 8.0 * x_dev[1]
        ci = T.zeros((3,))
        ci = T.set_subtensor(ci[0], x_dev[0] + 2.0 * x_dev[1] - 10.0)
        ci = T.set_subtensor(ci[1], x_dev[0])
        ci = T.set_subtensor(ci[2], x_dev[1])

        p = IPM(x0=x0, x_dev=x_dev, f=f, ci=ci, Ftol=Ftol, lbfgs=lbfgs,
                float_dtype=float_dtype, verbosity=verbosity)
        x, s, lda, fval, kkt = p.solve()

        gt = [float_dtype(4.0), float_dtype(3.0)]
        print('')
        print('Ground truth: [x, y] = [{}, {}]'.format(*gt))
        print('Solver solution: [x, y] = [{}, {}]'.format(*x))
        print('Slack variables: s = [{}, {}, {}]'.format(-s[0], s[1], s[2]))
        print(
            'Lagrange multipliers: lda = [{}, {}, {}]'.format(-lda[0], lda[1], lda[2]))
        print('f(x, y) = {}'.format(fval))
    elif prob == 6:
        print('Find the maximum entropy distribution of a six-sided die:')
        print(
            'maximize f(x) = -sum(x*log(x)) subject to sum(x) = 1 and x >= 0 (x.size == 6)')
        print('')
        x0 = np.random.rand(6).astype(float_dtype)
        x0 = x0 / np.sum(x0)

        f = T.sum(x_dev * T.log(x_dev + np.finfo(float_dtype).eps))
        ce = T.sum(x_dev) - 1.0
        ci = 1.0 * x_dev

        p = IPM(x0=x0, x_dev=x_dev, f=f, ce=ce, ci=ci, Ftol=Ftol, lbfgs=lbfgs, float_dtype=float_dtype,
                verbosity=verbosity)
        x, s, lda, fval, kkt = p.solve()

        gt = str(float_dtype(1.0 / 6.0))
        print('')
        print('Ground truth: [{}]'.format(', '.join([gt] * 6)))
        print('Solver solution: x = [{}]'.format(', '.join(list2str(x))))
        print('Slack variables: s = [{}]'.format(', '.join(list2str(s))))
        print('Lagrange multipliers: lda = [{}]'.format(
            ', '.join(list2str(lda))))
        print('f(x) = {}'.format(-fval))
    elif prob == 7:
        print(
            'maximize f(x, y, z) = x*y*z subject to x + y + z = 1, x >= 0, y >= 0, z >= 0')
        print('')
        x0 = np.random.randn(3).astype(float_dtype)

        f = -x_dev[0] * x_dev[1] * x_dev[2]
        ce = T.sum(x_dev) - 1.0
        ci = 1.0 * x_dev

        p = IPM(x0=x0, x_dev=x_dev, f=f, ce=ce, ci=ci, Ftol=Ftol, lbfgs=lbfgs, float_dtype=float_dtype,
                verbosity=verbosity)
        x, s, lda, fval, kkt = p.solve()

        gt = [float_dtype(1.0 / 3.0), float_dtype(1.0 / 3.0),
              float_dtype(1.0 / 3.0)]
        print('')
        print('Ground truth: [x, y, z] = [{}, {}, {}]'.format(*gt))
        print('Solver solution: [x, y, z] = [{}, {}, {}]'.format(*x))
        print('Slack variables: s = [{}, {}, {}]'.format(*s))
        print('Lagrange multipliers: lda = [{}, {}, {}, {}]'.format(*lda))
        print('f(x, y, z) = {}'.format(-fval))
    elif prob == 8:
        print('minimize f(x,y,z) = 4*x - 2*z subject to 2*x - y - z = 2, x**2 + y**2 = 1')
        print('')
        x0 = np.random.randn(3).astype(float_dtype)

        f = 4.0 * x_dev[1] - 2.0 * x_dev[2]
        ce = T.zeros((2,))
        ce = T.set_subtensor(ce[0], 2.0 * x_dev[0] - x_dev[1] - x_dev[2] - 2.0)
        ce = T.set_subtensor(ce[1], x_dev[0] ** 2 + x_dev[1] ** 2 - 1.0)

        p = IPM(x0=x0, x_dev=x_dev, f=f, ce=ce, Ftol=Ftol, lbfgs=lbfgs,
                float_dtype=float_dtype, verbosity=verbosity)
        x, s, lda, fval, kkt = p.solve()

        gt = [
            float_dtype(2.0 / np.sqrt(13.0)),
            float_dtype(-3.0 / np.sqrt(13.0)),
            float_dtype(-2.0 + 7.0 / np.sqrt(13.0))
        ]
        print('')
        print('Ground truth: [x, y, z] = [{}, {}, {}]'.format(*gt))
        print('Solver solution: [x, y, z] = [{}, {}, {}]'.format(*x))
        print('Lagrange multipliers: lda = [{}, {}]'.format(*lda))
        print('f(x, y, z) = {}'.format(fval))
    elif prob == 9:
        print(
            'minimize f(x, y) = (x - 2)**2 + 2*(y - 1)**2 subject to x + 4*y <= 3, x >= y')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)

        f = (x_dev[0] - 2.0) ** 2 + 2.0 * (x_dev[1] - 1.0) ** 2
        ci = T.zeros(2)
        ci = T.set_subtensor(ci[0], -x_dev[0] - 4.0 * x_dev[1] + 3.0)
        ci = T.set_subtensor(ci[1], x_dev[0] - x_dev[1])

        p = IPM(x0=x0, x_dev=x_dev, f=f, ci=ci, Ftol=Ftol, lbfgs=lbfgs,
                float_dtype=float_dtype, verbosity=verbosity)
        x, s, lda, fval, kkt = p.solve()

        gt = [float_dtype(5.0 / 3.0), float_dtype(1.0 / 3.0)]
        print('')
        print('Ground truth: [x, y] = [{}, {}]'.format(*gt))
        print('Solver solution: [x, y] = [{}, {}]'.format(*x))
        print('Slack variables: s = [{}, {}]'.format(-s[0], s[1]))
        print('Lagrange multipliers: lda = [{}, {}]'.format(-lda[0], lda[1]))
        print('f(x, y) = {}'.format(fval))
    elif prob == 10:
        print('minimize f(x, y, z) = (x - 1)**2 + 2*(y + 2)**2 + 3*(z + 3)**2 subject to z - y - x = 1, z - x**2 >= 0')
        print('')
        x0 = np.random.randn(3).astype(float_dtype)

        f = (x_dev[0] - 1.0) ** 2 + 2.0 * \
            (x_dev[1] + 2.0) ** 2 + 3.0 * (x_dev[2] + 3.0) ** 2
        ce = x_dev[2] - x_dev[1] - x_dev[0] - 1.0
        ci = x_dev[2] - x_dev[0] ** 2

        p = IPM(x0=x0, x_dev=x_dev, f=f, ce=ce, ci=ci, Ftol=Ftol, lbfgs=lbfgs, float_dtype=float_dtype,
                verbosity=verbosity)
        x, s, lda, fval, kkt = p.solve()

        gt = [0.12288, -1.1078, 0.015100]
        print('')
        print('Ground truth: [x, y, z] = [{}, {}, {}]'.format(*gt))
        print('Solver solution: [x, y, z] = [{}, {}, {}]'.format(*x))
        print('Slack variable: s = {}'.format(s))
        print('Lagrange multipliers: lda = [{}, {}]'.format(*lda))
        print('f(x, y) = {}'.format(fval))

    print('Karush-Kuhn-Tucker conditions (up to a sign):\n{}'.format(kkt))


if __name__ == '__main__':
    main()
