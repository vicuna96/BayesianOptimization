from main import *
import tensorflow as tf
import unittest

def np_ker(Xs, Zs, gamma):
    import numpy as np
    from scipy.spatial.distance import cdist
    return np.exp( - gamma * cdist(Xs.T, Zs.T ,metric='sqeuclidean'))

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



if __name__ == '__main__':
	unittest.main()