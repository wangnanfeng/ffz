import numpy as np
import matplotlib.pyplot as plt

numbers_a = np.array([1,2,3])
numbers_b = np.array([16,21,24])

#  y = 5 * x +10 
w ,b = 1, 2
m = numbers_a.shape[0]
plt.scatter(numbers_a,numbers_b)

def compute_cost(x,y,w,b):
    '''计算损失函数
    x : numbers_a
    y : numbers_b
    
    '''
    
    cost = 0
    
    for i in range(m):
        j = x[i] * w + b
        cost += (j - y[i])**2
    total_cost = 1/(2 * m)*cos
    
    return total_cost

def compute_gradient(x,y,w,b):
    "计算梯度"
    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        
        #dj_dw_i = 1/((x[i] * w + b -y[i]) * x[i]) * m
        j = x[i] * w + b
        
        dj_dw_i = (j - y[i]) * x[i]    
        dj_db_i = j - y[i]
        
        dj_dw += dj_dw_i
        dj_db += dj_db_i
        
             
    dj_dw = dj_dw / m
    dj_db = dj_db / m  
    
    return dj_dw,dj_db

def gradient_descent(x,y,w,b):
    
    for i in range(times):
        dj_dw,dj_db = compute_gradient(x,y,w,b)
        w -=  alpha * dj_dw
        b -= alpha * dj_db
    return w,b

alpha = 1.0e-2
times = 10000
w,b = gradient_descent(numbers_a,numbers_b,0,0)

x_values = np.linspace(0,3,100)
y_values = 5 * x_values +10 # 设想的直线
x_pre = np.linspace(0,3,100)
y_pre = w * x_pre + b # 计算出来的直线

# 绘图
plt.scatter(numbers_a,numbers_b)
plt.plot(x_values,y_values,label = 'real_function')
plt.plot(x_pre,y_pre,label = 'predict_function')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
