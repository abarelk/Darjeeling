
# coding: utf-8

import numpy as np


def sigmoid(Z):
    return 1. / (1. + np.exp(-Z))


def dsigmoid(Z):
    A = sigmoid(Z)
    return A * (1. - A)


def relu(Z):
    return np.maximum(0., Z)


def drelu(Z):
    return np.heaviside(Z, 0.)
    

def compute_cost(Yh, Y, parameters, lambd=0.):
    """
    Yh -- prediction vector of size(1, number of examples)
    Y -- label vector of size(1, number of examples)
    """
    m = Y.shape[1]
    L = len(parameters) // 2
    l2_regularization_cost = 0.
    for l in range(L):
        l2_regularization_cost = l2_regularization_cost + np.sum(np.square(parameters['W' + str(l+1)]))
    
    l2_regularization_cost = l2_regularization_cost * lambd / (2. * m)    
    cross_entropy_cost = np.sum((np.dot(Y, np.nan_to_num(np.log(Yh).T)) + np.dot((1. - Y), np.nan_to_num(np.log(1. - Yh).T))))/-m
        
    return cross_entropy_cost + l2_regularization_cost


def compute_dcost(Yh, Y):
    return -np.nan_to_num(np.divide(Y, Yh)) + np.nan_to_num(np.divide(1. - Y, 1. - Yh))
    

def initialize_parameters(layer_dims):
    parameters = {}
    
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters


def step_forward(W, b, Ap, activate):
    Z = np.dot(W, Ap) + b        
    return Z, activate(Z)
    

def linear_forward_propagation(parameters, X, activation): 
    A = X
    m = X.shape[1]
    
    caches = []
    L = len(parameters) // 2 + 1
    
    for l in range(1, L):       
        Ap = A
        Z, A = step_forward(parameters['W' + str(l)], 
                            parameters['b' + str(l)], 
                            Ap, 
                            sigmoid if l == L - 1 else activation)
        caches.append((Z, Ap, parameters['W' + str(l)], parameters['b' + str(l)]))
    
    return A, caches


def step_backward(cache, dA, dactivation, lambd=0):
    Z, Ap, W, b = cache
    m = Z.shape[1]
    
    dZ = dA * dactivation(Z)
    dW = np.dot(dZ, Ap.T)/m + lambd/m * W
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dAp = np.dot(W.T, dZ) 
    
    return dAp, dW, db

    
def linear_backward_propagation(caches, Yh, Y, dactivation, lambd=0):    
    L = len(caches)
    grads = {}
    
    current_cache = caches[L-1]
    dAL = compute_dcost(Yh, Y)
    grads['dA' + str(L)] = dAL
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = step_backward(current_cache, dAL, dsigmoid, lambd)
    
    dA = grads['dA' + str(L-1)]
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA' + str(l)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = step_backward(current_cache, dA, dactivation, lambd) 
        dA = grads['dA' + str(l)]
        
    return grads


def predict(X, y, parameters, activation):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    
    # Forward propagation
    a3, caches = linear_forward_propagation(parameters, X, activation)
    
    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p


def initialize_adam_variables(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

    return v, s


def update_parameters(parameters, grads, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    
    v_crted = {}
    s_crted = {}
    for l in range(L):
        W_idx = "W" + str(l+1)
        b_idx = "b" + str(l+1)
        dW_idx = "d" + W_idx
        db_idx = "d" + b_idx
        
        # momentum
        v[dW_idx] = beta1 * v[dW_idx] + (1-beta1) * grads[dW_idx]    # exponentially weighted average
        v[db_idx] = beta1 * v[db_idx] + (1-beta1) * grads[db_idx]
        
        v_crted[dW_idx] = np.divide(v[dW_idx], 1 - beta1**t)         # to adjust early means 
        v_crted[db_idx] = np.divide(v[db_idx], 1 - beta1**t)
        
        # RMSprop
        s[dW_idx] = beta2 * s[dW_idx] + (1-beta2) * np.square(grads[dW_idx])
        s[db_idx] = beta2 * s[db_idx] + (1-beta2) * np.square(grads[db_idx])
        s_crted[dW_idx] = np.divide(s[dW_idx], 1 - beta2**t)
        s_crted[db_idx] = np.divide(s[db_idx], 1 - beta2**t)

        parameters[W_idx] = parameters[W_idx] - learning_rate * np.divide(v_crted[dW_idx], np.sqrt(s_crted[dW_idx]) + epsilon)
        parameters[b_idx] = parameters[b_idx] - learning_rate * np.divide(v_crted[db_idx], np.sqrt(s_crted[db_idx]) + epsilon)

    return parameters, v, s


def random_mini_batches(X, Y, mini_batch_size=0, seed=0):   
    m = X.shape[1]
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    
    if mini_batch_size == 0:
        mini_batch_size = m
        
    start = 0
    end = mini_batch_size
    batches = []
    while end < m:
        batches.append((shuffled_X[:, start:end], shuffled_Y[:, start:end]))
        start = end
        end += mini_batch_size
    
    if end >= m:
        batches.append((shuffled_X[:, start:m], shuffled_Y[:, start:m]))

    return batches


def initialize_hyper_params(hyper_params):
    layer_dims = hyper_params['layer_dims'] if 'layer_dims' in hyper_params else None
    mini_batch_size = hyper_params['mini_batch_size'] if 'mini_batch_size' in hyper_params else 0
    learning_rate = hyper_params['learning_rate'] if 'learning_rate' in hyper_params else 0.7
    epochs = hyper_params['epochs'] if 'epochs' in hyper_params else 30000
    l2_lambda = hyper_params['l2_lambda'] if 'l2_lambda' in hyper_params else 0
    
    return layer_dims, mini_batch_size, learning_rate, epochs, l2_lambda


"""
X: data (features, number of examples)
Y: label (1, number of examples)
hyperparameters
    layer_dims
    mini_batch_size
    learning_rate
    epochs
    l2_lambda
seed: random seed for test
debug_print_iter: print per each iter#
"""
def gradient_descent(X, Y, hyper_params, debug_print_iter=10000, seed=0):
    """
    X -- data of size (n, number of examples)
    Y -- true "label" vector of size (1, number of examples)
    layer_dims -- the number of nodes in layers including X
    """
    layer_dims, mini_batch_size, learning_rate, epochs, lambd = initialize_hyper_params(hyper_params)
    
    parameters = initialize_parameters(layer_dims)
    v, s = initialize_adam_variables(parameters)
    
    for i in range(epochs):        
        seed = seed + 1             
        batches = random_mini_batches(X, Y, mini_batch_size, seed)
                       
        for mini_batch_X, mini_batch_Y in batches:            
            Yh, caches = linear_forward_propagation(parameters, mini_batch_X, relu)
            grads = linear_backward_propagation(caches, Yh, mini_batch_Y, drelu, lambd)
                
            cost = compute_cost(Yh, mini_batch_Y, parameters, lambd)
            parameters, v, s = update_parameters(parameters, grads, v, s, i+1, learning_rate)

        if (i % debug_print_iter == 0):
            print("Cost after iteration {}: {}".format(i, cost))            

    return parameters, cost


def _to_theta_vector(dic, layers, wparam, bparam):
    L = layers    
    keys = []
    theta = None
    layer_dims = [dic[wparam + '1'].shape[1]]
    
    for l in range(1, L + 1):
        vw = dic[wparam + str(l)].reshape((-1, 1))
        vb = dic[bparam + str(l)].reshape((-1, 1))
        keys = keys + [wparam + str(l)]*vw.shape[0]
        keys = keys + [bparam + str(l)]*vb.shape[0]
        
        layer_dims = layer_dims + [dic[wparam + str(l)].shape[0]]
        if (l == 1):
            theta = np.append(vw, vb, axis=0)
        else:
            theta = np.append(theta, np.append(vw, vb, axis=0), axis=0)
        
    return theta, keys, layer_dims


def parameters_to_vector(parameters):
    return _to_theta_vector(parameters, len(parameters) // 2, 'W', 'b')


def gradients_to_vector(grads):
    return _to_theta_vector(grads, len(grads) // 3, 'dW', 'db')
    

def vector_to_parameters(theta, layer_dims):
    parameters = initialize_parameters(layer_dims)
    wstart = 0
    for l in range(1, len(layer_dims)):        
        wend = wstart + layer_dims[l] * layer_dims[l-1]
        bstart = wend
        bend = bstart + layer_dims[l] 
        parameters['W' + str(l)] = theta[wstart:wend, 0].reshape((layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = theta[bstart:bend, 0].reshape((layer_dims[l], 1))
        wstart = bend

    return parameters

        
def gradient_check(X, Y, activation, parameters, grads, epsilon=1e-7, lambd=0.):    
    theta, _, layer_dims = parameters_to_vector(parameters)    
    gradapprox = np.zeros((theta.shape[0], 1))
    # Loop for every single element in parameters (eg. W11, W12, W13, W21, ...)
    for idx in range(theta.shape[0]):
        theta_p = np.copy(theta)
        theta_p[idx, 0] = theta_p[idx, 0] + epsilon
        theta_m = np.copy(theta)
        theta_m[idx, 0] = theta_m[idx, 0] - epsilon
        
        params_p = vector_to_parameters(theta_p, layer_dims)
        Yh, caches = linear_forward_propagation(params_p, X, activation)
        J_p = compute_cost(Yh, Y, params_p, lambd)
        
        params_m = vector_to_parameters(theta_m, layer_dims)
        Yh, caches = linear_forward_propagation(params_m, X, activation)
        J_m = compute_cost(Yh, Y, params_m, lambd)

        gradapprox[idx] = (J_p - J_m) / (2. * epsilon)
    
    grads_v, _, _ = gradients_to_vector(grads)
    difference = np.linalg.norm(grads_v - gradapprox) / (np.linalg.norm(grads_v) + np.linalg.norm(gradapprox))
    if difference > 2e-7:
        print("There might be a mistake in backward propagation. difference: {}, grads_v: {}, gradapprox: {}".format(difference, grads_v, gradapprox))
    else:
        print("Backward propagation works correctly. difference: {}".format(difference))
    
    return difference


def accuracy(parameters, X, Y):
    p, caches = linear_forward_propagation(parameters, X, relu)
    return 100. - np.sum(np.abs(p - Y))/Y.shape[1] * 100

