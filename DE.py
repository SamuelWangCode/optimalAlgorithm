import numpy as np
import random
from test_function import test_function_names
from test_function import test_functions
from test_function import value_ranges
from limit_variables import *


class DE:
    def __init__(self, function_num=0, dim=2, iter_max=50, swarm_size=50, CR=0.2, F=0.5):
        self.swarm_size = swarm_size
        self.function_num = function_num
        self.dim = dim
        self.CR = CR
        self.F = F
        self.X = np.zeros((swarm_size, dim))
        self.fun = np.ones((swarm_size))
        self.solution = np.zeros((dim))
        self.global_params = [0 for x in range(dim)]
        self.global_opt = float("inf")
        self.iter_num = 0
        self.eval_count = 0
        self.iter_max = iter_max
        self.function_name = test_function_names[function_num]
        self.value_range = value_ranges[function_num]
        random.seed()

    def stopping_condition(self):
        status = bool(self.iter_num >= self.iter_max)
        return status

    def memory_best_value(self):
        for i in range(self.swarm_size):
            if (self.fun[i] < self.global_opt):
                self.global_opt = np.copy(self.fun[i])
                self.global_params = np.copy(self.X[i][:])

    def obj_function(self, X):
        return test_functions[self.function_num](X, self.dim)

    def init_introduction(self):
        print('DE初始化完成，测试函数为'+str(self.function_name)+'，维数为'+str(self.dim) +
              '，使用粒子数为'+str(self.swarm_size)+'，将进行'+str(self.iter_max)+'次迭代。')
        print('----------------------')
        print('初始化粒子的最优位置为：')
        print(self.global_params)
        print('初始化粒子的最优函数值为：')
        print(self.global_opt)

    def iter_introduction(self):
        print('-------第'+str(self.iter_num+1)+'次迭代--------')
        print('粒子的最优位置为：')
        print(self.global_params)
        print('粒子的最优函数值为：')
        print(self.global_opt)

    def end_introduction(self):
        print('满足终止条件，迭代结束')
        print('粒子的最优位置为：')
        print(self.global_params)
        print('粒子的最优函数值为：')
        print(self.global_opt)
        print('一共计算了'+str(self.eval_count)+'次函数值')

    def increase_iter_num(self):
        self.iter_num += 1

    def init_swarm(self):
        self.eval_count = 0
        self.iter_num = 0
        for i in range(self.swarm_size):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(
                    self.value_range[0], self.value_range[1])
            self.fun[i] = self.obj_function(self.X[i])
            self.eval_count = self.eval_count + 1
            if self.fun[i] < self.global_opt:
                self.global_opt = self.fun[i]
                self.global_params = np.copy(self.X[i][:])

    def differential(self):
        u = np.copy(self.X)
        for i in range(self.swarm_size):
            a = list(range(self.swarm_size))
            a.remove(i)
            random_arr = random.sample(a, 3)
            select_x = [self.X[random_arr[0]],
                        self.X[random_arr[1]], self.X[random_arr[2]]]
            u[i] = select_x[0]+self.F*(select_x[1]-select_x[2])
            u[i] = limit_variables(u[i], self.value_range)
        v = np.copy(self.X)
        for i in range(self.swarm_size):
            drand = random.sample(range(self.dim), 1)
            for j in range(self.dim):
                if random.random() < self.CR or j == drand[0]:
                    v[i][j] = u[i][j]
            aff = self.obj_function(v[i])
            self.eval_count += 1
            if aff < self.fun[i]:
                self.X[i] = np.copy(v[i])
                self.fun[i] = aff
        self.memory_best_value()

    def get_DE(self):
        self.init_swarm()
        # self.init_introduction()
        while(not(self.stopping_condition())):
            self.differential()
            # self.iter_introduction()
            self.increase_iter_num()
        # self.end_introduction()
