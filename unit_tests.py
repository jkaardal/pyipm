import theano
import theano.tensor as T
import numpy as np
import math
from pyipm import IPM
import time

float_dtype = np.float64
verbosity = -1
hess_type = [False, 4]
Ftol = 1.0E-8
Stol = 1.0E-3

x_dev = T.vector('x_dev')
lambda_dev = T.vector('lda_dev')

print "Testing unconstrained problems..."

text_abbr = ["f", "df", "d2f", "ce", "dce", "d2ce", "ci", "dci", "d2ci"]
text_width = 34
state_types = ["precompiled", "expression", "auto-diff"]

def make_str(state, comp, m):
    string = []
    for i in range(len(state)):
        if state[i] == comp and (not m or not text_abbr[i].startswith("d2")):
            string.append(text_abbr[i])   
    return ",".join(string)

def make_text_state(state, m):
    string = []
    for comp in state_types:
        substring = make_str(state, comp, m)
        string.append(substring.center(text_width))
    return "|".join(string) 

state_blacklist = []
state_blacklist.append(["NULL", None, None, None, None, None, None, None, None])
state_blacklist.append(["auto-diff", None, None, None, None, None, None, None, None])
state_blacklist.append(["precompiled", "auto-diff", None, None, None, None, None, None, None])
state_blacklist.append(["precompiled", None, "auto-diff", None, None, None, None, None, None])
state_blacklist.append(["precompiled", "NULL", None, None, None, None, None, None, None])
state_blacklist.append(["expression", "NULL", None, None, None, None, None, None, None])
state_blacklist.append([None, None, None, "auto-diff", None, None, None, None, None])
state_blacklist.append([None, None, None, "precompiled", "auto-diff", None, None, None, None])
state_blacklist.append([None, None, None, "precompiled", None, "auto-diff", None, None, None])
state_blacklist.append([None, None, None, None, None, None, "precompiled", "auto-diff", None])
state_blacklist.append([None, None, None, None, None, None, "precompiled", None, "auto-diff"])
state_blacklist.append([None, None, None, None, None, None, "auto-diff", None, None])
state_blacklist.append([None, None, None, "NULL", "precompiled", None, None, None, None])
state_blacklist.append([None, None, None, "NULL", "expression", None, None, None, None])
state_blacklist.append([None, None, None, "NULL", "auto-diff", None, None, None, None])
state_blacklist.append([None, None, None, "NULL", None, "precompiled", None, None, None])
state_blacklist.append([None, None, None, "NULL", None, "expression", None, None, None])
state_blacklist.append([None, None, None, "NULL", None, "auto-diff", None, None, None])
state_blacklist.append([None, None, None, None, None, None, "NULL", "precompiled", None])
state_blacklist.append([None, None, None, None, None, None, "NULL", "expression", None])
state_blacklist.append([None, None, None, None, None, None, "NULL", "auto-diff", None])
state_blacklist.append([None, None, None, None, None, None, "NULL", None, "precompiled"])
state_blacklist.append([None, None, None, None, None, None, "NULL", None, "expression"])
state_blacklist.append([None, None, None, None, None, None, "NULL", None, "auto-diff"])
state_blacklist.append([None, None, None, "precompiled", "NULL", None, None, None, None])
state_blacklist.append([None, None, None, "precompiled", None, "NULL", None, None, None])
state_blacklist.append([None, None, None, "expression", "NULL", None, None, None, None])
state_blacklist.append([None, None, None, "expression", None, "NULL", None, None, None])
state_blacklist.append([None, None, None, None, None, None, "precompiled", "NULL", None])
state_blacklist.append([None, None, None, None, None, None, "precompiled", None, "NULL"])
state_blacklist.append([None, None, None, None, None, None, "expression", "NULL", None])
state_blacklist.append([None, None, None, None, None, None, "expression", None, "NULL"])

test_problems = []

p1 = dict()
p1["text_statements"] = ["minimize f(x,y) = x**2 - 4*x + y**2 - y - x*y"]
p1["f"] = x_dev[0]**2 - 4*x_dev[0] + x_dev[1]**2 - x_dev[1] - x_dev[0]*x_dev[1]
p1["ce"] = None
p1["neq"] = 0
p1["ci"] = None
p1["nineq"] = 0
p1["init"] = np.random.randn(2).astype(float_dtype)
p1["ground_truth"] = [np.array([3.0, 2.0], dtype=float_dtype)]

test_problems.append(p1)

p2 = dict()
p2["text_statements"] = ["Find the global minimum of the 2D Rosenbrock function.", "minimize f(x,y) = 100*(y-x**2)**2 + (1-x)**2"]
p2["f"] = 100*(x_dev[1]-x_dev[0]**2)**2 + (1-x_dev[0])**2
p2["ce"] = None
p2["neq"] = 0
p2["ci"] = None
p2["nineq"] = 0
p2["init"] = np.random.randn(2).astype(float_dtype)
p2["ground_truth"] = [np.array([1.0, 1.0], dtype=float_dtype)]

test_problems.append(p2)

p3 = dict()
p3["text_statements"] = ["maximize f(x,y) = x+y subject to x**2 + y**2 = 1"]
p3["f"] = -T.sum(x_dev)
p3["ce"] = T.sum(x_dev ** 2)-1.0
p3["neq"] = 1
p3["ci"] = None
p3["nineq"] = 0
p3["init"] = np.random.randn(2).astype(float_dtype)
p3["ground_truth"] = [np.array([np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0], dtype=float_dtype)]

test_problems.append(p3)

p4 = dict()
p4["text_statements"] = ["maximize f(x,y) = (x**2)*y subject to x**2 + y**2 = 3"]
p4["f"] = -(x_dev[0]**2)*x_dev[1]
p4["ce"] = T.sum(x_dev ** 2) - 3.0
p4["neq"] = 1
p4["ci"] = None
p4["nineq"] = 0
p4["init"] = np.random.randn(2).astype(float_dtype)
p4["ground_truth"] = [np.array([np.sqrt(2.0), 1.0], dtype=float_dtype), np.array([-np.sqrt(2.0), 1.0], dtype=float_dtype), np.array([0.0, -np.sqrt(3)], dtype=float_dtype)]

test_problems.append(p4)

p5 = dict()
p5["text_statements"] = ["minimize f(x,y) = x**2 + 2*y**2 + 2*x + 8*y subject to -x-2*y+10 <= 0, x >= 0, y >= 0"]
p5["f"] = x_dev[0]**2 + 2.0*x_dev[1]**2 + 2.0*x_dev[0] + 8.0*x_dev[1]
p5["ce"] = None
p5["neq"] = 0
ci = T.zeros((3,))
ci = T.set_subtensor(ci[0], x_dev[0]+2.0*x_dev[1]-10.0)
ci = T.set_subtensor(ci[1], x_dev[0])
ci = T.set_subtensor(ci[2], x_dev[1])
p5["ci"] = ci
p5["nineq"] = 3
p5["init"] = np.random.randn(2).astype(float_dtype)
p5["ground_truth"] = [np.array([4.0, 3.0], dtype=float_dtype)]

test_problems.append(p5)

p6 = dict()
p6["text_statements"] = ["Find the maximum entropy distribution of a six-sided die:", "maximize f(x) = -sum(x*log(x)) subject to sum(x) = 1 and x >= 0 (x.size=6)"]
p6["f"] = T.sum(x_dev*T.log(x_dev + np.finfo(float_dtype).eps))
p6["ce"] = T.sum(x_dev) - 1.0
p6["neq"] = 1
p6["ci"] = 1.0*x_dev
p6["nineq"] = 6
p6["init"] = np.random.rand(6).astype(float_dtype)
p6["ground_truth"] = [np.array([1.0/6.0]*6, dtype=float_dtype)]

test_problems.append(p6)

p7 = dict()
p7["text_statements"] = ["maximize f(x,y,z) = x*y*z subject to x+y+z = 1, x >= 0, y >= 0, z >= 0"]
p7["f"] = -x_dev[0]*x_dev[1]*x_dev[2]
p7["ce"] = T.sum(x_dev) - 1.0
p7["neq"] = 1
p7["ci"] = 1.0*x_dev
p7["nineq"] = 3
p7["init"] = np.random.randn(3).astype(float_dtype)
p7["ground_truth"] = [np.array([1.0/3.0]*3, dtype=float_dtype)]

test_problems.append(p7)

p8 = dict()
p8["text_statements"] = ["minimize f(x,y,z) = 4*x-2*z subject to 2*x-y-z = 2, x**2 + y**2 = 1"]
p8["f"] = 4.0*x_dev[1] - 2.0*x_dev[2]
ce = T.zeros((2,))
ce = T.set_subtensor(ce[0], 2.0*x_dev[0]-x_dev[1]-x_dev[2]-2.0)
ce = T.set_subtensor(ce[1], x_dev[0]**2+x_dev[1]**2-1.0)
p8["ce"] = ce
p8["neq"] = 2
p8["ci"] = None
p8["nineq"] = 0
p8["init"] = np.random.randn(3).astype(float_dtype)
p8["ground_truth"] = [np.array([2.0/np.sqrt(13.0), -3.0/np.sqrt(13.0), -2.0+7.0/np.sqrt(13.0)], dtype=float_dtype)]

test_problems.append(p8)

p9 = dict()
p9["text_statements"] = ["minimize f(x,y) = (x-2)**2 + 2*(y-1)**2 subject to x+4*y <= 3, x >= y"]
p9["f"] = (x_dev[0]-2.0)**2 + 2.0*(x_dev[1]-1.0)**2
p9["ce"] = None
p9["neq"] = 0
ci = T.zeros(2)
ci = T.set_subtensor(ci[0], -x_dev[0]-4.0*x_dev[1]+3.0)
ci = T.set_subtensor(ci[1], x_dev[0]-x_dev[1])
p9["ci"] = ci
p9["nineq"] = 2
p9["init"] = np.random.randn(2).astype(float_dtype)
p9["ground_truth"] = [np.array([5.0/3.0, 1.0/3.0], dtype=float_dtype)]

test_problems.append(p9)

p10 = dict()
p10["text_statements"] = ["minimize f(x,y,z) = (x-1)**2 + 2*(y+2)**2 + 3*(z+3)**2 subject to z-y-x = 1, z-x**2 >= 0"]
p10["f"] = (x_dev[0]-1.0)**2 + 2.0*(x_dev[1]+2.0)**2 + 3.0*(x_dev[2]+3.0)**2
p10["ce"] = x_dev[2] - x_dev[1] - x_dev[0] - 1.0
p10["neq"] = 1
p10["ci"] = x_dev[2] - x_dev[0]**2
p10["nineq"] = 1
p10["init"] = np.random.randn(3).astype(float_dtype)
p10["ground_truth"] = [np.array([0.12288, -1.1078, 0.015100], dtype=float_dtype)]

test_problems.append(p10)

header = [x.center(text_width) for x in state_types]
header = "|".join(header)
breakline = "-"*(len(state_types)*text_width)

test_results = []

none_state_types = ["NULL"]
none_state_types.extend(state_types)
max_idx = len(none_state_types)-1

for lbfgs in hess_type:
    print breakline
    if lbfgs:
        print "USING L-BFGS WITH lbfgs=" + str(lbfgs)
    else:
        print "USING EXACT HESSIAN"
    print breakline

    idx = [0]*len(text_abbr)
    done = False
    while not done:
        f_state = none_state_types[idx[0]]
        df_state = none_state_types[idx[1]]
        d2f_state = none_state_types[idx[2]]
        ce_state = none_state_types[idx[3]]
        dce_state = none_state_types[idx[4]]
        d2ce_state = none_state_types[idx[5]]
        ci_state = none_state_types[idx[6]]
        dci_state = none_state_types[idx[7]]
        d2ci_state = none_state_types[idx[8]]

        state = [f_state, df_state, d2f_state, ce_state, dce_state, d2ce_state, ci_state, dci_state, d2ci_state]

        # check blacklist
        match = False
        for i in range(len(state_blacklist)):
            blist = state_blacklist[i]
            for j in range(len(blist)):
                if state[j] == blist[j]:
                    match = True
                elif blist[j] is not None:
                    match = False
                    break
            if match:
                break

        #if lbfgs and (d2f_state == "NULL" and d2ce_state == "NULL" and d2ci_state == "NULL"):
        #    print state

        #if lbfgs and not (d2f_state == "NULL" and d2ce_state == "NULL" and d2ci_state == "NULL"):
        #    match = True
        if not lbfgs and (d2f_state == "NULL" or (ce_state != "NULL" and d2ce_state == "NULL") or (ci_state != "NULL" and d2ci_state == "NULL")):
            match = True

        if not match:
            print header
            print breakline
            print make_text_state(state, lbfgs)
            print breakline

            for j in range(len(test_problems)):
                problem = test_problems[j]

                if state[3] != "NULL" and problem["ce"] is None:
                    continue
                if state[6] != "NULL" and problem["ci"] is None:
                    continue
                if state[3] == "NULL" and problem["ce"] is not None:
                    continue
                if state[6] == "NULL" and problem["ci"] is not None:
                    continue

                statements = problem["text_statements"]
                for k in range(len(statements)):
                    print statements[k]

                x0 = problem["init"]

                f = problem["f"]

                if state[1] == "auto-diff":
                    df = None
                elif state[1] == "expression":
                    df = T.grad(f, x_dev)
                else:
                    df = T.grad(f, x_dev)
                    df = theano.function(inputs=[x_dev], outputs=df)

                if lbfgs:
                    d2f = None
                else:
                    if state[2] == "auto-diff":
                        d2f = None
                    elif state[2] == "expression":
                        d2f = theano.gradient.hessian(cost=f, wrt=x_dev)
                    else:
                        d2f = theano.gradient.hessian(cost=f, wrt=x_dev)
                        d2f = theano.function(inputs=[x_dev], outputs=d2f)

                if state[0] == "precompiled":
                    f = theano.function(inputs=[x_dev], outputs=f)

                if problem["ce"] is not None:
                    ce = problem["ce"]

                    if state[4] == "auto-diff":
                        dce = None
                    elif state[4] == "expression":
                        dce = theano.gradient.jacobian(ce, wrt=x_dev).reshape((problem["neq"], x0.size)).T
                    else:
                        dce = theano.gradient.jacobian(ce, wrt=x_dev).reshape((problem["neq"], x0.size)).T
                        dce = theano.function(inputs=[x_dev], outputs=dce)

                    if lbfgs:
                        d2ce = None
                    else:
                        if state[5] == "auto-diff":
                            d2ce = None
                        elif state[5] == "expression":
                            d2ce = theano.gradient.hessian(cost=T.sum(ce*lambda_dev[:problem["neq"]]), wrt=x_dev)
                        else:
                            d2ce = theano.gradient.hessian(cost=T.sum(ce*lambda_dev[:problem["neq"]]), wrt=x_dev)
                            d2ce = theano.function(inputs=[x_dev, lambda_dev], outputs=d2ce)

                    if state[3] == "precompiled":
                        ce = theano.function(inputs=[x_dev], outputs=ce)
                else:
                    ce = None
                    dce = None
                    d2ce = None

                if problem["ci"] is not None:
                    ci = problem["ci"]

                    if state[7] == "auto-diff":
                        dci = None
                    elif state[7] == "expression":
                        dci = theano.gradient.jacobian(ci, wrt=x_dev).reshape((problem["nineq"], x0.size)).T
                    else:
                        dci = theano.gradient.jacobian(ci, wrt=x_dev).reshape((problem["nineq"], x0.size)).T
                        dci = theano.function(inputs=[x_dev], outputs=dci)

                    if lbfgs:
                        d2ci = None
                    else:
                        if state[8] == "auto-diff":
                            d2ci = None
                        elif state[8] == "expression":
                            d2ci = theano.gradient.hessian(cost=T.sum(ci*lambda_dev[problem["neq"]:]), wrt=x_dev)
                        else:
                            d2ci = theano.gradient.hessian(cost=T.sum(ci*lambda_dev[problem["neq"]:]), wrt=x_dev)
                            d2ci = theano.function(inputs=[x_dev, lambda_dev], outputs=d2ci)

                    if state[6] == "precompiled":
                        ci = theano.function(inputs=[x_dev], outputs=ci)
                else:
                    ci = None
                    dci = None
                    d2ci = None

                p = IPM(x0=x0, x_dev=x_dev, f=f, df=df, d2f=d2f, ce=ce, dce=dce, d2ce=d2ce, ci=ci, dci=dci, d2ci=d2ci, lambda_dev=lambda_dev, Ftol=Ftol, lbfgs=lbfgs, float_dtype=float_dtype, verbosity=verbosity)
                x, s, lda, fval, kkt = p.solve()

                converge = False
                for x_gt in problem["ground_truth"]:
                    if np.linalg.norm(x_gt-x) <= Stol:
                        converge = True
                        break

                if not converge:
                    print x_gt
                    print x
                    test_results.append(False)
                    raise Exception("FAILED!")
                else:
                    test_results.append(True)
                    print "PASSED!"
                print ""
        
        i = 0
        idx[i] += 1
        while True:
            if idx[i] > max_idx and i+1 < len(idx):
                idx[i] = 0
                idx[i+1] += 1
                i += 1
            elif idx[i] >= len(none_state_types) and i+1 >= len(idx):
                done = True
                break
            else:
                break

if all(test_results):
    print "ALL TESTS PASSED (" + str(len(test_results)) + " passed, 0 failed)"
else:
    num_passed = sum(test_results)
    print "SOME TESTS FAILED (" + str(num_passed) + " passed, " + str(len(test_results)-num_passed) + " failed)"
