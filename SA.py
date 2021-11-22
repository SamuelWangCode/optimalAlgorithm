import numpy as np
import random
from test_function import test_function_names
from test_function import test_functions
from test_function import value_ranges
from limit_variables import *


class SA:
    def __init__(self, function_num=0, dim=2, iter_max=50, T_max=100, T_min=1, mean_markov=50, step_factor=0.4):
        self.function_num = function_num
        self.dim = dim
        self.T_max = T_max
        self.T_min = T_min
        self.T = T_max
        self.iter_max = iter_max
        self.mean_markov = mean_markov
        self.step_factor = step_factor
        self.fun = float("inf")
        self.solution = np.zeros((dim))
        self.global_params = [0 for x in range(dim)]
        self.global_opt = float("inf")
        self.iter_num = 0
        self.eval_count = 0
        self.function_name = test_function_names[function_num]
        self.value_range = value_ranges[function_num]
        random.seed()

    def stopping_condition(self):
        status = bool(self.iter_num >= self.iter_max)
        return status

    def obj_function(self, X):
        return test_functions[self.function_num](X)

    def init_introduction(self):
        print('初始化完成，测试函数为'+str(self.function_name)+'，维数为'+str(self.dim))
        print('----------------------')
        print('初始化粒子的位置为：')
        print(self.global_params)
        print('初始化粒子的函数值为：')
        print(self.global_opt)

    def iter_introduction(self):
        print('-------第'+str(self.iter_num+1)+'次迭代--------')
        print('粒子的当前位置为：')
        print(self.solution)
        print('粒子的当前函数值为：')
        print(self.fun)
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

    def init(self):
        self.eval_count = 0
        self.iter_num = 0
        for i in range(self.dim):
            self.solution[i] = random.uniform(
                self.value_range[0], self.value_range[1])
        self.fun = self.obj_function(self.solution)
        self.eval_count = self.eval_count + 1
        self.global_opt = self.fun
        self.global_params = np.copy(self.solution)

    def generate(self):
        t = np.random.randint(2)
        new_x = np.copy(self.solution)
        new_x[t] = self.solution[t] + self.step_factor * \
            self.value_range[t] * (random.random()-0.5) * 2
        limit_variables(new_x, self.value_range)
        return new_x

    def accept_value(self, new_x):
        new_y = self.obj_function(new_x)
        self.eval_count += 1
        if new_y < self.fun:
            self.solution = new_x
            self.fun = new_y
            if new_y < self.global_opt:
                self.global_opt = new_y
                self.global_params = new_x
        else:
            accept_rate = np.exp(-(new_y-self.fun)/self.T)
            reference_rate = random.random()
            if accept_rate >= reference_rate:
                self.solution = new_x
                self.fun = new_y

    def lower_t(self):
        result = self.T_max - (self.T_max - self.T_min) / \
            self.iter_max * self.iter_num
        self.T = result

    def get_SA(self):
        self.init()
        self.init_introduction()
        while(not(self.stopping_condition())):
            for _ in range(self.mean_markov):
                x = self.generate()
                self.accept_value(x)
            self.iter_introduction()
            self.increase_iter_num()
            self.lower_t()
        self.end_introduction()
