import numpy as np
import math
import random
from test_function import test_function_names
from test_function import test_functions
from test_function import value_ranges
from limit_variables import *


class CS:
    def __init__(self, function_num=0, dim=2, iter_max=50, swarm_size=25, lamuda=1, beta=1.5, pa=0.25):
        self.swarm_size = swarm_size
        self.function_num = function_num
        self.dim = dim
        self.lamuda = lamuda
        self.beta = beta
        self.pa = pa
        self.X = np.zeros((swarm_size, dim))
        self.fun = np.ones((swarm_size))
        self.solution = np.zeros((dim))
        self.global_params = [0 for x in range(dim)]
        self.global_opt = float("inf")
        self.iter_num = 0
        self.eval_count = 0
        self.global_pos = 0
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
                self.global_pos = i

    def obj_function(self, X):
        return test_functions[self.function_num](X, self.dim)

    def init_introduction(self):
        print('CS初始化完成，测试函数为'+str(self.function_name)+'，维数为'+str(self.dim) +
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
                self.global_pos = i

    def get_new_nest_via_levy(self):
        sigma_u = (math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / (
            math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2)))) ** (1 / self.beta)
        sigma_v = 1
        Xt = np.copy(self.X)
        for i in range(self.swarm_size):
            s = self.X[i, :]
            u = np.random.normal(0, sigma_u, 1)
            v = np.random.normal(0, sigma_v, 1)
            Ls = u / ((abs(v)) ** (1 / self.beta))
            # lamuda的设置关系到点的活力程度  方向是由最佳位置确定的  有点类似PSO算法  但是步长不一样
            stepsize = self.lamuda*Ls*(s-self.global_params)
            s = s + stepsize * np.random.randn(1, len(s))  # 产生满足正态分布的序列
            Xt[i, :] = s
            limit_variables(Xt[i], self.value_range)
        return Xt

    def empty_nests(self):
        ori_nest = np.copy(self.X)
        ori_nest = np.delete(ori_nest,self.global_pos,0)
        nest1 = np.copy(ori_nest)
        nest2 = np.copy(ori_nest)
        rand_m = self.pa - np.random.rand(self.swarm_size-1,self.dim)
        rand_m = np.heaviside(rand_m,0)
        np.random.shuffle(nest1)
        np.random.shuffle(nest2)
        stepsize = np.random.rand(1,1) * (nest1 - nest2)
        new_nest = ori_nest + stepsize * rand_m
        new_nest = np.append(new_nest, [self.X[self.global_pos]], axis=0)
        for i in range(self.swarm_size):
            limit_variables(new_nest[i],self.value_range)
        return new_nest
        

    def get_best_nest(self, newnest):
        for i in range(self.swarm_size):
            temp = self.obj_function(newnest[i, :])
            self.eval_count += 1
            if self.fun[i] > temp:
                self.X[i] = newnest[i]
                self.fun[i] = temp
                if temp < self.global_opt:
                    self.global_opt = temp
                    self.global_params = self.X[i, :]
                    self.global_pos = i

    def get_CS(self):
        self.init_swarm()
        # self.init_introduction()
        while(not(self.stopping_condition())):
            newnest = self.get_new_nest_via_levy()
            self.get_best_nest(newnest)
            nest_c = self.empty_nests()
            self.get_best_nest(nest_c)
            # self.iter_introduction()
            self.increase_iter_num()
        # self.end_introduction()
