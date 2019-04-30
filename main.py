#!/usr/bin/env python3
import os
import numpy as np
from numpy import random
import scipy
from scipy import special
import matplotlib
import mnist
import pickle
import math
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.animation as animation

from tqdm import tqdm
#from scipy.special import softmax

import tensorflow as tf
sess = tf.Session()

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

### hyperparameter settings and other constants
### end hyperparameter settings

def load_MNIST_dataset_with_validation_split():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        np.random.seed(8675309)
        perm = np.random.permutation(60000)
        Xs_tr = np.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = np.ascontiguousarray(Ys_tr[:,perm])
        # extract out a validation set
        Xs_va = Xs_tr[:,50000:60000]
        Ys_va = Ys_tr[:,50000:60000]
        Xs_tr = Xs_tr[:,1:50000]
        Ys_tr = Ys_tr[:,1:50000]
        # load test data
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset

# compute the cumulative distribution function of a standard Gaussian random variable
def gaussian_cdf(u):
    return 0.5*(1.0 + tf.math.erf(u/np.sqrt(2.0)))

# compute the probability mass function of a standard Gaussian random variable
def gaussian_pmf(u):
    return tf.math.exp(-u**2/2.0)/np.math.sqrt(2.0*np.pi)


# compute the Gaussian RBF kernel matrix for a vector of data points (in TensorFlow)
#
# Xs        points at which to compute the kernel (size: d x m)
# Zs        other points at which to compute the kernel (size: d x n)
# gamma     gamma parameter for the RBF kernel
#
# returns   an (m x n) matrix Sigma where Sigma[i,j] = K(Xs[:,i], Zs[:,j])
def rbf_kernel_matrix(Xs, Zs, gamma):
    # TODO students should implement this
    m, n = tf.shape(Xs)[1], tf.shape(Zs)[1]
    inner = - 2 * tf.matmul(tf.transpose(Xs), Zs)
    dist = tf.reshape(tf.reduce_sum(Xs ** 2, axis=0), [m, 1]) + tf.reshape(tf.reduce_sum(Zs ** 2, axis=0), [1, n])
    return tf.exp( - gamma * (dist + inner))


# compute the distribution predicted by a Gaussian process that uses an RBF kernel (in TensorFlow)
#
# Xs            points at which to compute the kernel (size: d x n) where d is the number of parameters
# Ys            observed value at those points (size: n)
# gamma         gamma parameter for the RBF kernel
# sigma2_noise  the variance sigma^2 of the additive gaussian noise used in the model
#
# returns   a function that takes a value Xtest (size: d) and returns a tuple (mean, variance)
def gp_prediction(Xs, Ys, gamma, sigma2_noise):
    # first, do any work that can be shared among predictions
    # TODO students should implement this
    sigma = rbf_kernel_matrix(Xs, Xs, gamma) + tf.cast(sigma2_noise * tf.eye(tf.shape(Xs)[1]), tf.float64)

    toMat = lambda x: tf.reshape(x, [-1, 1])

    inv = tf.linalg.inv(sigma)
    inv_y = tf.matmul(inv, toMat(Ys))
    # next, define a nested function to return
    def prediction_mean_and_variance(Xtest):
        # TODO students should implement this
        k_vec = lambda x_star : rbf_kernel_matrix(Xs, toMat(x_star), gamma)
        mean = tf.matmul(tf.transpose(k_vec(Xtest)), inv_y)
        quad = lambda x_star : tf.matmul(tf.transpose(k_vec(x_star)), tf.matmul(inv, k_vec(x_star))) + sigma2_noise
        variance = rbf_kernel_matrix( toMat(Xtest), toMat(Xtest), gamma ) - quad(Xtest)
        # construct mean and variance
        return (mean[0,0], variance[0,0])
    #finally, return the nested function
    return prediction_mean_and_variance


# compute the probability of improvement (PI) acquisition function
#
# Ybest     points at which to compute the kernel (size: d x n)
# mean      mean of prediction
# stdev     standard deviation of prediction (the square root of the variance)
#
# returns   PI acquisition function
def pi_acquisition(Ybest, mean, stdev):
    # TODO students should implement this
    return gaussian_cdf( (Ybest - mean) / stdev)


# compute the expected improvement (EI) acquisition function
#
# Ybest     points at which to compute the kernel (size: d x n)
# mean      mean of prediction
# stdev     standard deviation of prediction
#
# returns   EI acquisition function
def ei_acquisition(Ybest, mean, stdev):
    # TODO students should implement this
    Ynormalized = (Ybest - mean) / stdev
    return - (gaussian_pmf(Ynormalized) + Ynormalized * gaussian_cdf(Ynormalized) ) * stdev


# return a function that computes the lower confidence bound (LCB) acquisition function
#
# kappa     parameter for LCB
#
# returns   function that computes the LCB acquisition function
def lcb_acquisition(kappa):
    def A_lcb(Ybest, mean, stdev):
        return mean - kappa * stdev
        # TODO students should implement this
    return A_lcb


# gradient descent to do the inner optimization step of Bayesian optimization
#
# objective     the objective function to minimize, as a function that takes a tensorflow variable and returns an expression
# d             the dimension of the input
# alpha         learning rate/step size
# num_iters     number of iterations of gradient descent      
#
# returns       a function that takes input
#   x0            initial value to assign to variable x
#               and runs gradient descent, and 
#   returns     (obj_min, x_min), where
#       obj_min     the value of the objective after running iterations of gradient descent
#       x_min       the value of x after running iterations of gradient descent
def gradient_descent(objective, d, alpha, num_iters):
    # construct the tensorflow graph associated with this objective
    x = tf.Variable(np.zeros((d,)))
    f = objective(x)
    (g, ) = tf.gradients(f, [x])
    sess.run(tf.global_variables_initializer())
    gd_step = x.assign(x - alpha * g)
    # a function that computes gradient descent
    def gd_from_initial_value(x0):
        with sess.as_default():
            x.assign(x0).eval()
            for it in range(num_iters):
                gd_step.eval()
            return (f.eval().item(), x.eval())
    return gd_from_initial_value


# run Bayesian optimization to minimize an objective
#
# objective     objective function
# d             dimension to optimize over
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# acquisition   acquisition function to use (e.g. ei_acquisition)
# random_x      function that returns a random sample of the parameter we're optimizing over (e.g. for use in warmup)
# gd_nruns      number of random initializations we should use for gradient descent for the inner optimization step
# gd_alpha      learning rate for gradient descent
# gd_niters     number of iterations for gradient descent
# n_warmup      number of initial warmup evaluations of the objective to use
# num_iters     number of outer iterations of Bayes optimization to run (including warmup)
#s
# returns       tuple of (y_best, x_best, Ys, Xs), where
#   y_best          objective value of best point found
#   x_best          best point found
#   Ys              vector of objective values for all points searched (size: num_iters)
#   Xs              matrix of all points searched (size: d x num_iters)
def bayes_opt(objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters):
    # TODO students should implement this
    y_best = np.inf
    x_best = random_x()
    Xs, Ys = np.zeros((d, num_iters)), np.zeros((num_iters))
    for iter in range(n_warmup):
        x = random_x()
        Xs[:, iter] = x
        y = objective(x)
        Ys[iter] = y
        if y <= y_best:
            x_best, y_best = x, y
        print("iteration", iter)
    for iter in range(n_warmup, num_iters):
        lmbd = gp_prediction(Xs[:,:iter], Ys[:iter], gamma, sigma2_noise)
        def inner_objective(x_test):
            mean, variance = lmbd(x_test)
            return acquisition(y_best, mean, variance)
        grad_des = gradient_descent(inner_objective, d, gd_alpha, gd_niters)
        sess.run(tf.global_variables_initializer())
        x_inner, y_inner = [], []
        with sess.as_default():
            for _ in range(gd_nruns):
                x = random_x()
                some_y, some_x = grad_des(x)
                x_inner.append(some_x)
                y_inner.append(objective(x_inner[-1]))
        ind = np.argmin(y_inner)
        x, y = x_inner[ind], y_inner[ind]
        Xs[:, iter], Ys[iter] = x, y
        if y <= y_best:
            x_best, y_best = x, y
        print("iteration", iter)
        print(y_best, x_best)

    return (y_best, x_best, Ys, Xs)

# a one-dimensional test objective function on which to run Bayesian optimization
def test_objective(x):
    return (np.cos(8.0*x) - 0.3 + (x-0.5)**2)


# produce an animation of the predictions made by the Gaussian process in the course of 1-d Bayesian optimization
#
# objective     objective function
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# Ys            vector of objective values for all points searched (size: num_iters)
# Xs            matrix of all points searched (size: d x num_iters)
# xs_eval       list of xs at which to evaluate the mean and variance of the prediction at each step of the algorithm
# filename      path at which to store .mp4 output file 
def animate_predictions(objective, gamma, sigma2_noise, Ys, Xs, xs_eval, filename):
    mean_eval = []
    variance_eval = []
    # Set up formatting for the movie files
    for it in range(len(Ys)):
        print("rendering frame %i" % it)
        Xsi = Xs[:, 0:(it+1)]
        Ysi = Ys[0:(it+1)]
        gp_pred = gp_prediction(Xsi, Ysi, gamma, sigma2_noise)
        pred_means = []
        pred_variances = []
        XE = tf.Variable(np.zeros((1,)))
        (pred_mean, pred_variance) = gp_pred(XE)
        with sess.as_default():
            for x_eval in xs_eval:
                XE.assign(np.array([x_eval])).eval()
                pred_means.append(pred_mean.eval().item())
                pred_variances.append(pred_variance.eval().item())
        mean_eval.append(np.array(pred_means))
        variance_eval.append(np.array(pred_variances))

    fig, ax = pyplot.subplots()

    def anim_init():
        fig.clear()

    def animate(i):
        ax = fig.gca()
        ax.clear()
        ax.fill_between(xs_eval, mean_eval[i] + 2.0*np.sqrt(variance_eval[i]), mean_eval[i] - 2.0*np.sqrt(variance_eval[i]), color="#eaf1f7")
        ax.plot(xs_eval, objective(xs_eval))
        ax.plot(xs_eval, mean_eval[i], color="r")
        ax.scatter(Xs[0,0:(i+1)], Ys[0:(i+1)])
        pyplot.title("Bayes Opt After %d Steps" % (i+1))
        pyplot.xlabel("parameter")
        pyplot.ylabel("objective")
        fig.show()

    ani = animation.FuncAnimation(fig, animate, frames=range(len(Ys)), init_func=anim_init, interval=400, repeat_delay=1000)

    ani.save(filename)


# compute the gradient of the multinomial logistic regression objective, with regularization (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should use their implementation from programming assignment 2
    Xs, Ys = Xs[:,ii], Ys[:,ii]
    ewx = np.exp(np.matmul(W,Xs))
    p = np.sum(ewx, axis=0)
    return 1 / Xs.shape[1] * np.matmul(-Ys+ 1/p * ewx, Xs.T) + gamma * W


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should use their implementation from programming assignment 1
    return np.mean(np.argmax(np.matmul(W, Xs), axis=0) != np.argmax(Ys, axis=0))


# compute the cross-entropy loss of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    # TODO students should use their implementation from programming assignment 3
    ewx = np.exp(np.matmul(W, Xs))
    logSoft = np.log(ewx / np.sum(ewx, axis=0))
    return - np.sum(Ys * logSoft) / Xs.shape[1] + gamma / 2 * np.sum(W ** 2)


# SGD + Momentum: run stochastic gradient descent with minibatching, sequential sampling order, and momentum (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    models = []
    V0 = 0 * np.zeros(W0.shape)
    for i in range(num_epochs):
        cur = i*(n//B)
        for j in range(n // B):
            ii = np.arange(j*B,(j+1)*B)
            V0 = beta * V0 - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0)
            W0 = W0 + V0
            if (j+cur+1) % monitor_period == 0:
                models.append(W0)
        #print("epoch", i)
    return models


# produce a function that runs SGD+Momentum on the MNIST dataset, initializing the weights to zero
#
# mnist_dataset         the MNIST dataset, as returned by load_MNIST_dataset_with_validation_split
# num_epochs            number of epochs to run for
# B                     the batch size
#
# returns               a function that takes parameters
#   params                  a numpy vector of shape (3,) with entries that determine the hyperparameters, where
#       gamma = 10^(-8 * params[0])
#       alpha = 0.5*params[1]
#       beta = params[2]
#                       and returns (the validation error of the final trained model after all the epochs) minus 0.9.
#                       if training diverged (i.e. any of the weights are non-finite) then return 0.1, which corresponds to an error of 1.
def mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B):
    # TODO students should implement this
    Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te = mnist_dataset

    c, _ = Ys_tr.shape
    d, _ = Xs_tr.shape
    W0 = np.zeros((c, d))

    def train(params):
        models= sgd_mss_with_momentum(Xs_tr, Ys_tr, 10**(-8 * params[0]), W0, 0.5 * params[1], params[2], B,
                                      num_epochs, Xs_tr.shape[1] // B)
        print("Number of models trained", len(models))
        if np.isinf(models[-1]).any() or np.isnan(models[-1]).any():
            return 0.1
        else:
            return multinomial_logreg_error(Xs_va, Ys_va, models[-1]) - .9

    return train


if __name__ == "__main__":
    # TODO students should implement plotting functions here
    import timeit
    from argparse import ArgumentParser

    # Define the parser to run script on cmd
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--part1", action='store_true',
                        help="To run part 1 of the assignment")
    parser.add_argument("--part2", action='store_true',
                        help="To run part 2 of the assignment")
    parser.add_argument("--part3", action='store_true',
                        help="To run part 3 of the assignment")
    parser.add_argument("--time", action='store_true',
                        help="To run the time experiment part of the assignment")

    args = parser.parse_args()

    if args.part1:
        d = 1
        random_x =  lambda : np.array([np.random.uniform()])

        gamma, sigma2_noise = 10, 0.001
        gd_alpha, gd_nruns, gd_niters = 0.01, 5, 100
        n_warmup, num_iters = 3, 20

        optima = []
        acquis = [pi_acquisition, ei_acquisition, lcb_acquisition(2)]
        names = ["pi", "ei", "lcb"]
        for acquisition in acquis:
            y_best, x_best, Ys, Xs = bayes_opt(test_objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
            optima.append( x_best )

        f = open("part1.txt",'a')
        for name, x in zip(names, optima):
            f.write(name+" "+str(x)+'\n')
        f.closed

    if args.part2:        
        y_best, x_best, Ys, Xs = bayes_opt(test_objective, d, gamma, sigma2_noise, pi_acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
        mini, maxi = np.min(Xs), np.max(Xs)
        xs_eval = np.arange(mini,maxi, (maxi-mini)/15)
        animate_predictions(test_objective, gamma, sigma2_noise, Ys, Xs, xs_eval, "animation0.mp4")

    if args.part3:
        B, num_epochs = 600, 5
        ker_gamma = 10
        sigma2_noise = 0.001
        kappa = 2

        gd_alpha, gd_nruns, gd_niters = 0.05, 5, 100
        n_warmup, num_iters = 3, 20

        d = 3
        random_x =  lambda : np.random.uniform(size=3)
        
        mnist_dataset = load_MNIST_dataset_with_validation_split()
        objective = mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B)
        y_best, x_best, Ys, Xs = bayes_opt(objective, d, ker_gamma, sigma2_noise, lcb_acquisition(kappa), random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
        print("Best x", x_best)
        print("Best Y", y_best)

        f = open("part3a_lcb.txt",'a')
        for x, y in zip(Xs, Ys):
            f.write( str(x) + " " + str(y) + '\n')
        f.write("Best parameters " + str(x_best) + " " + str(y_best) + '\n')
        f.closed

    if args.time:
        pass
        
