import numpy as np
import random
from test_function import test_function_names
from test_function import test_functions
from test_function import value_ranges
from limit_variables import *


class PSO:
    def __init__(self, function_num=0, iter_max=50, swarm_size=50, dim=2, w_max=0.2, w_min=0.5, c1=2, c2=2):
        self.swarm_size = swarm_size
        self.function_num = function_num
        self.dim = dim
        self.w_max = w_max
        self.w_min = w_min
        self.w = w_max
        self.c1 = c1
        self.c2 = c2
        self.X = np.zeros((swarm_size, dim))
        self.V = np.zeros((swarm_size, dim))
        self.p_best = np.zeros((swarm_size, dim))
        self.p_aff = np.zeros(swarm_size)
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
        return test_functions[self.function_num](X)

    def init_introduction(self):
        print('初始化完成，测试函数为'+str(self.function_name)+'，维数为'+str(self.dim) +
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
            self.p_best[i] = np.copy(self.X[i])
            self.p_aff[i] = self.fun[i]
            if self.fun[i] < self.global_opt:
                self.global_opt = self.fun[i]
                self.global_params = np.copy(self.X[i][:])

    def change_w(self):
        self.w = self.w_max - ((self.w_max - self.w_min) /
                               self.iter_max) * self.iter_num

    def cal_pso(self):
        for p in range(self.swarm_size):
            for q in range(self.dim):
                self.V[p][q] = self.w * self.V[p][q] + self.c1 * random.random() * (self.p_best[p][q] -
                                                                                    self.X[p][q]) + self.c2 * random.random() * (self.global_params[q] - self.X[p][q])
                self.X[p][q] = self.X[p][q] + self.V[p][q]
            self.V[p] = limit_variables(self.V[p], self.value_range)
            self.X[p] = limit_variables(self.X[p], self.value_range)
            aff = self.obj_function(self.X[p])
            self.eval_count += 1
            if aff < self.p_aff[p]:
                self.p_aff[p] = aff
                self.p_best[p] = np.copy(self.X[p])
            if aff < self.global_opt:
                self.global_opt = aff
                self.global_params = np.copy(self.X[p])

    def get_PSO(self):
        self.init_swarm()
        self.init_introduction()
        while(not(self.stopping_condition())):
            self.cal_pso()
            self.change_w()
            self.iter_introduction()
            self.increase_iter_num()
        self.end_introduction()
