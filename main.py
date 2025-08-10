import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
# from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('ggplot')
plt.show()

# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

print("x_train:", x_train)
print("y_train:", y_train)


def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost = 0 

    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = cost + (f_wb - y[i])**2  

    total_cost = (1 / (2 * m)) * cost  

    return total_cost


def compute_gradient(x, y, w, b):
    """
    Computes the gradient of the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        dw (float): The gradient with respect to w
        db (float): The gradient with respect to b
    """
    m = x.shape[0]  # number of training examples
    
    dw = 0.0
    db = 0.0
    
    for i in range(m):
        f_wb = w * x[i] + b
        dw += (f_wb - y[i]) * x[i]
        db += (f_wb - y[i])
    
    dw /= m
    db /= m
    
    return dw, db


def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    """
    Performs gradient descent to learn w and b.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w_init (float): Initial value of w
      b_init (float): Initial value of b
      alpha (float): Learning rate
      num_iters (int): Number of iterations for gradient descent
    
    Returns
        w (float): Final value of w after gradient descent
        b (float): Final value of b after gradient descent
    """
    w = w_init
    b = b_init
    
    for i in range(num_iters):
        dw, db = compute_gradient(x, y, w, b)
        w -= alpha * dw
        b -= alpha * db
    
    return w, b


w_init = 0.0
b_init = 0.0
# alpha = 0.01
alpha = 1.0e-2
num_iters = 10000
w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, alpha, num_iters)
print("Final w:", w_final) 
print("Final b:", b_final)

# gradients = compute_gradient(x_train, y_train, 200.0, 100.0)
# print("Gradient at w=200, b=100: ", compute_gradient(x_train, y_train, 200.0, 100.0))
# result = compute_cost(x_train, y_train, 200.0, 100.0)
# print("Cost at w=200, b=100: ", result)