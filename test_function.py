import numpy as np  


def Ackley(X, d=2):
    a = 20
    b = 0.2 
    c = 2 * np.pi
    temp1 = 0
    temp2 = 0
    for i in range(d):
        temp1 = temp1 + X[i]**2
        temp2 = temp2 + np.cos(c * X[i])
    result = -a*np.exp(-b*np.sqrt(1/d*temp1))-np.exp(1/d*temp2)+a+np.exp(1)
    return result
    

def Griewank(X, d=2):
    temp1 = 0
    temp2 = 1
    for i in range(d):
        temp1 = temp1 + X[i]**2/4000
        temp2 = temp2 * np.cos(X[i]/np.sqrt(i+1))
    result = temp1 - temp2 + 1
    return result


def LevyF(X, d=2):
    w = [0 for j in range(d)]
    temp1 = 0
    for i in range(d):
        w[i] = 1 + (X[i] - 1) / 4
        if i < d - 1:
            temp1 = temp1 + (w[i]-1)**2*(1+10*np.sin(np.pi*w[i]+1)**2)
    result = np.sin(np.pi*w[0])**2+temp1+(w[d-1]-1)**2*(1+np.sin(2*np.pi*w[d-1])**2)
    return result


def Rastrigin(X, d=2):
    result = 10 * d
    for i in range(d):
        result = result + X[i]**2 - 10*np.cos(2*np.pi*X[i])
    return result


def Schwefel(X, d=2):
    temp1 = 0
    for i in range(d):
        temp1 = temp1 + X[i]*np.sin(np.sqrt(np.abs(X[i])))
    result = 418.9829 * d - temp1
    return result


def Sphere(X, d=2):
    result = 0
    for i in range(d):
        result = result + X[i]**2
    return result


def Zakharov(X, d=2):
    temp1 = 0
    temp2 = 0
    for i in range(d):
        temp1 = temp1 + X[i]**2
        temp2 = temp2 + 0.5 * (i+1) * X[i]
    result = temp1 + temp2**2 + temp2**4
    return result


def Rosenbrock(X, d=2):
    result = 0
    for i in range(d):
        if i < d-1:
            result = result + 100*(X[i+1]-X[i]**2)**2 + (X[i]-1)**2
    return result


def Michalewicz(X, d=2):
    result = 0
    m = 10
    for i in range(d):
        result = result - np.sin(X[i])*np.sin((i+1)*X[i]**2/np.pi)**(2*m)
    return result


def Styblinski_Tang(X, d=2):
    temp = 0
    for i in range(d):
        temp = temp + X[i]**4 - 16*X[i]**2 + 5*X[i]
    result = temp / 2
    return result


test_functions = [Ackley, Griewank, LevyF, Rastrigin, Schwefel, Sphere, Zakharov, Rosenbrock, Michalewicz, Styblinski_Tang]
test_function_names = ['Ackley', 'Griewank', 'LevyF', 'Rastrigin', 'Schwefel', 'Sphere', 'Zakharov', 'Rosenbrock', 'Michalewicz', 'Styblinski_Tang']
test_function_min = [0,0,0,0,0,0,0,0,'?','-39.16599*d']
value_ranges = [[-32.768, 32.768],[-600, 600],[-10, 10],[-5.12, 5.12],[-500, 500],[-5.12, 5.12],[-5.12, 5.12],[-5, 10],[0, np.pi],[-5, 5]]