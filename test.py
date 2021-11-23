from DE import *
from ABC import *
from CS import *
from PSO import *
from SA import *
from CSA import *
import numpy as np
from matplotlib import pyplot as plt
from test_function import *

fun = np.zeros((5,10,6,30))
for i in range(5):
    print(test_function_names[i])
    for j in range(10):
        print('dim=',j+1)
        a = DE(i,j+1)
        b = ABC(i,j+1)
        c = CS(i,j+1)
        d = PSO(i,j+1)
        e = SA(i,j+1)
        f = CSA(i,j+1)
        for p in range(30):
            a.get_DE()
            b.get_ABC()
            c.get_CS()
            d.get_PSO()
            e.get_SA()
            f.get_CSA()
            fun[i][j][0][p] = a.global_opt
            fun[i][j][1][p] = b.global_opt
            fun[i][j][2][p] = c.global_opt
            fun[i][j][3][p] = d.global_opt
            fun[i][j][4][p] = e.global_opt
            fun[i][j][5][p] = f.global_opt

print(fun)

np.savez('./fun.npz', fun = fun)












