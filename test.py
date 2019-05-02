from main import *
import tensorflow as tf
import unittest

def np_ker(Xs, Zs, gamma):
    import numpy as np
    from scipy.spatial.distance import cdist
    return np.exp( - gamma * cdist(Xs.T, Zs.T ,metric='sqeuclidean'))

def np_gp_pred(Xs, Ys, gamma, sigma2_noise):
    to_mat = lambda x : x.reshape(-1,1)

    sigma = np_ker(Xs, Xs, gamma) + sigma2_noise * np.eye(Xs.shape[1])
    inv_sig = np.linalg.inv(sigma)

    k_star = lambda x_star : np_ker(Xs, to_mat(x_star), gamma)

    mean = lambda x_star : np.matmul(np.matmul(k_star(x_star).T, inv_sig), to_mat(Ys))[0,0]

    first  = lambda x_star : np_ker(to_mat(x_star), to_mat(x_star), gamma) + sigma2_noise
    quad = lambda x_star :  np.matmul(np.matmul(k_star(x_star).T), k_star(x_star) )
    var = lambda x_star : first(x_star)[0,0] + quad(x_star)[0,0]

    return (mean, var)


# initialize a session
sess = tf.Session()

# define the tolerance
atol = 1e-06
# define the class for test cases
class TestImageProcessing(unittest.TestCase):
    def test_kernel(self):
        rand_var_1 = tf.Variable(tf.random_uniform([5, 4], 0, 10, dtype=tf.float32))
        rand_var_2 = tf.Variable(tf.random_uniform([5, 3], 0, 10, dtype=tf.float32))
        rand_gamma = tf.Variable(tf.random_uniform([1], 0, 1, dtype=tf.float32))

        tfker = rbf_kernel_matrix(rand_var_1, rand_var_2, rand_gamma)

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()
        # initialize variables
        sess.run(init_op)
        # get variables out of the session
        # sess.run([rand_var_1, rand_var_2, rand_gamma])

        npker = np_ker(rand_var_1.eval(session=sess), rand_var_2.eval(session=sess), rand_gamma.eval(session=sess))
        tfker = rbf_kernel_matrix(rand_var_1, rand_var_2, rand_gamma)

        self.assertTrue(np.allclose(npker, tfker.eval(session=sess), atol=atol))

    def test_gp_pred(self):
        d, n = 11, 22
        preds = 7

        rand_var_1 = tf.Variable(tf.random_uniform([d, n], 0, 10, dtype=tf.float32))
        rand_var_2 = tf.Variable(tf.random_uniform([n], 0, 10, dtype=tf.float32))
        rand_gamma = tf.Variable(tf.random_uniform([1], 0, 1, dtype=tf.float32))
        rand_sigma = tf.Variable(tf.random_uniform([1], 0, 1, dtype=tf.float32)) * 1e-3

        new_points = tf.Variable(tf.random_uniform([d, preds], 0, 10, dtype=tf.float32))

        tf_gp = lambda xtest : gp_prediction(rand_var_1, rand_var_2, rand_gamma, rand_sigma)(xtest)

        predicts = [tf_gp(new_points[:,i] for i in range(preds))]

        Xs, Ys, gamma, sigma, tf_predicts, points = sess.run(
            [rand_var_1, rand_var_2, rand_gamma, rand_sigma, predicts, new_points])

        tf_means = np.array([pred[0] for pred in tf_predicts])
        tf_vars = np.array([pred[1] for pred in tf_predicts])

        np_gd = lambda xtest : np_gp_pred(Xs, Ys, gamma, sigma)(xtest)

        np_predicts = [np_gd(points[:,i] for i in range(preds))]

        np_means = np.array([pred[0] for pred in np_predicts])
        np_vars = np.array([pred[1] for pred in np_predicts])

        self.assertTrue(np.allclose(tf_means, np_means))
        self.assertTrue(np.allclose(tf_vars, np_vars))



if __name__ == '__main__':
	unittest.main()