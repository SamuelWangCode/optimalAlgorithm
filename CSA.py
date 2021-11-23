import numpy as np
import random
import heapq
from test_function import test_function_names
from test_function import test_functions
from test_function import value_ranges
from limit_variables import *


class CSA:
    def __init__(self, function_num=0, dim=2, iter_max=100, swarm_size=100, selection_size=5, max_clone=10, mutation_rate=0.3, mutation_step=1, drop_size=20):
        self.swarm_size = swarm_size
        self.function_num = function_num
        self.dim = dim
        self.selection_size = selection_size
        self.max_clone = max_clone
        self.mutation_rate = mutation_rate
        self.mutation_step = mutation_step
        self.drop_size = drop_size
        self.X = np.zeros((swarm_size, dim))
        self.fun = np.ones((swarm_size))
        self.aff = np.zeros((swarm_size))
        self.solution = np.zeros((dim))
        self.global_params = [0 for x in range(dim)]
        self.global_opt = float("inf")
        self.best_index_list = np.zeros((selection_size))
        self.best_fun_list = np.zeros((selection_size))
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
        print('CSA初始化完成，测试函数为'+str(self.function_name)+'，维数为'+str(self.dim) +
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

    def calculate_aff(self, fun):
        if fun > 0:
            aff = 1/(1 + fun)
        else:
            aff = 1
        return aff 

    def init_swarm(self):
        self.eval_count = 0
        self.iter_num = 0
        for i in range(self.swarm_size):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(
                    self.value_range[0], self.value_range[1])
            self.fun[i] = self.obj_function(self.X[i])
            self.eval_count = self.eval_count + 1
            self.aff[i] = self.calculate_aff(self.fun[i])
            if self.fun[i] < self.global_opt:
                self.global_opt = self.fun[i]
                self.global_params = np.copy(self.X[i][:])

    def select(self):
        self.best_fun_list = heapq.nsmallest(self.selection_size, self.fun)
        self.best_index_list = list(map(self.fun.tolist().index, heapq.nsmallest(self.selection_size, self.fun)))

    def clone(self):
        clone_cells = []
        fun_arr = []
        aff_arr = []
        for i in range(self.selection_size):
            index = self.best_index_list[i]
            cell = self.X[index]
            fun = self.fun[index]
            aff = self.aff[index]
            clone_num = round(max(1,self.aff[index]*self.max_clone))
            for _ in range(clone_num):
                clone_cells.append(np.copy(cell))
                fun_arr.append(fun)
                aff_arr.append(aff)
        return clone_cells, fun_arr, aff_arr

    def mutation(self, clone_cells, fun_arr, aff_arr):
        mutationed_cells = []
        rdn = np.random.random(len(clone_cells))
        for i in range(len(clone_cells)):
            if rdn[i] < self.mutation_rate:
                cell = np.copy(clone_cells[i])
                j = np.random.randint(self.dim)
                cell[j] = cell[j] + (random.random()*2-1)*self.mutation_step*self.value_range[1]
                cell = limit_variables(cell, self.value_range)
                mutationed_cells.append(cell)
                fun_arr[i] = self.obj_function(cell)
                aff_arr[i] = self.calculate_aff(fun_arr[i])
                self.eval_count += 1
            else:
                mutationed_cells.append(np.copy(clone_cells[i]))
        return mutationed_cells, fun_arr, aff_arr

    def regroup(self, mutationed_cells, fun_arr, aff_arr):
        self.X = np.vstack((self.X,mutationed_cells))
        self.fun = np.append(self.fun,fun_arr)
        self.aff = np.append(self.aff,aff_arr)

    def reselect(self):
        remain_index_list = list(map(self.fun.tolist().index, heapq.nsmallest(self.swarm_size - self.drop_size, self.fun)))
        self.X = self.X.take(remain_index_list, 0)
        self.fun = self.fun.take(remain_index_list)
        self.aff = self.aff.take(remain_index_list)
        self.global_opt = self.fun[0]
        self.global_params = self.X[0]

    def reinit(self):
        pos = np.zeros((self.drop_size, self.dim))
        fun = np.zeros((self.drop_size))
        aff = np.zeros((self.drop_size))
        for i in range(self.drop_size):
            for j in range(self.dim):
                pos[i][j] = random.uniform(
                    self.value_range[0], self.value_range[1])
            fun[i] = self.obj_function(pos[i])
            self.eval_count += 1
            aff[i] = self.calculate_aff(fun[i])
            if fun[i] < self.global_opt:
                self.global_opt = fun[i]
                self.global_params = np.copy(pos[i])
        self.X = np.vstack((self.X,pos))
        self.fun = np.append(self.fun,fun)
        self.aff = np.append(self.aff,aff)

    def get_CSA(self):
        self.init_swarm()
        # self.init_introduction()
        while(not(self.stopping_condition())):
            self.select()
            clone_cells, fun_arr, aff_arr = self.clone()
            mutationed_cells, fun_arr, aff_arr = self.mutation(clone_cells, fun_arr, aff_arr)
            self.regroup(mutationed_cells, fun_arr, aff_arr)
            self.reselect()
            self.reinit()
            # self.iter_introduction()
            self.increase_iter_num()
        # self.end_introduction()
