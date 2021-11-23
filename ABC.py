import numpy as np
import random
from test_function import test_function_names
from test_function import test_functions
from test_function import value_ranges
from limit_variables import *


class ABC:
    def __init__(self, function_num=0, dim=2, iter_max=50, swarm_size=25, trail_max=20):
        self.swarm_size = swarm_size
        self.function_num = function_num
        self.dim = dim
        self.X = np.zeros((swarm_size, dim))
        self.fun = np.ones((swarm_size))
        self.trial = np.zeros((swarm_size))
        self.solution = np.zeros((dim))
        self.prob = [0 for x in range(swarm_size)]
        self.global_params = [0 for x in range(dim)]
        self.global_opt = float("inf")
        self.iter_num = 0
        self.eval_count = 0
        self.iter_max = iter_max
        self.trial_max = trail_max
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
        print('ABC初始化完成，测试函数为'+str(self.function_name)+'，维数为'+str(self.dim) +
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

    def init(self, index):
        for i in range(self.dim):
            self.X[index][i] = random.uniform(
                    self.value_range[0], self.value_range[1])
            self.solution = np.copy(self.X[index][:])
            self.fun[index] = self.obj_function(self.solution)
            self.trial[index] = 0

    def send_employed_bees(self):
        i = 0
        while (i < self.swarm_size) and (not (self.stopping_condition())):
            r = random.random()
            self.param2change = (int)(r * self.dim)

            r = random.random()
            self.neighbour = (int)(r * self.swarm_size)
            while self.neighbour == i:
                r = random.random()
                self.neighbour = (int)(r * self.swarm_size)
            self.solution = np.copy(self.X[i][:])

            r = random.random()
            self.solution[self.param2change] = self.X[i][self.param2change] + (
                self.X[i][self.param2change] - self.X[self.neighbour][self.param2change]) * (
                r - 0.5) * 2
            limit_variables(self.solution, self.value_range)
            self.ObjValSol = self.obj_function(self.solution)
            self.eval_count += 1
            if (self.ObjValSol < self.fun[i]):
                self.trial[i] = 0
                self.X[i][:] = np.copy(self.solution)
                self.fun[i] = self.ObjValSol
            else:
                self.trial[i] = self.trial[i] + 1
            i += 1

    def calculate_probabilities(self):
        minf = np.copy(min(self.fun))
        for i in range(self.swarm_size):
            self.prob[i] = (0.9 * ((1/(1+self.fun[i])) / (1/(1+minf)))) + 0.1

    def send_onlooker_bees(self):
        i = 0
        t = 0
        while (t < self.swarm_size) and (not (self.stopping_condition())):
            r = random.random()
            if ((r < self.prob[i]) or (r > self.prob[i])):
                t += 1
                r = random.random()
                self.param2change = (int)(r * self.dim)
                r = random.random()
                self.neighbour = (int)(r * self.swarm_size)
                while self.neighbour == i:
                    r = random.random()
                    self.neighbour = (int)(r * self.swarm_size)
                self.solution = np.copy(self.X[i][:])

                r = random.random()
                self.solution[self.param2change] = self.X[i][self.param2change] + (
                    self.X[i][self.param2change] - self.X[self.neighbour][self.param2change]) * (
                    r - 0.5) * 2
                limit_variables(self.solution, self.value_range)
                self.ObjValSol = self.obj_function(self.solution)
                self.eval_count += 1
                if (self.ObjValSol < self.fun[i]):
                    self.trial[i] = 0
                    self.X[i][:] = np.copy(self.solution)
                    self.fun[i] = self.ObjValSol
                else:
                    self.trial[i] = self.trial[i] + 1
            i += 1
            i = i % self.swarm_size

    def send_scout_bees(self):
        if np.amax(self.trial) >= self.trial_max:
            self.init(self.trial.argmax(axis = 0))

    def get_ABC(self):
        self.init_swarm()
        # self.init_introduction()
        while(not(self.stopping_condition())):
            self.send_employed_bees()
            self.calculate_probabilities()
            self.send_onlooker_bees()
            self.memory_best_value()
            self.send_scout_bees()
            # self.iter_introduction()
            self.increase_iter_num()
        self.end_introduction()
