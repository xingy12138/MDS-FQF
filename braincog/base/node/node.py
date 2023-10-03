# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/4/10 18:46
# User      : Floyed
# Product   : PyCharm
# Project   : braincog
# File      : node.py
# explain   : 缂佷胶鍋熺划锟犲礂閸愵厼螡闁绘劕婀辩悮顐﹀垂閿燂拷1闁跨噦鎷�7

import abc
import math
from abc import ABC
import numpy as np
import random
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from einops import rearrange, repeat
from braincog.base.strategy.surrogate import *
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

class BaseNode(nn.Module, abc.ABC):

    def __init__(self,
                 threshold=.5,
                 v_reset=0.,
                 dt=1.,
                 step=8,
                 requires_thres_grad=False,
                 sigmoid_thres=False,
                 requires_fp=False,
                 layer_by_layer=False,
                 n_groups=1,
                 *args,
                 **kwargs):

        super(BaseNode, self).__init__()
        #self.threshold = Parameter(torch.tensor(threshold), requires_grad=requires_thres_grad)
        self.threshold = torch.tensor(threshold, requires_grad=False)
        self.sigmoid_thres = sigmoid_thres
        self.mem = 0.
        self.spike = 0.
        self.dt = dt
        self.feature_map = []
        self.mem_collect = []
        self.requires_fp = requires_fp
        self.v_reset = v_reset
        self.step = step
        self.layer_by_layer = layer_by_layer
        self.groups = n_groups
        self.mem_detach = kwargs['mem_detach'] if 'mem_detach' in kwargs else False
        self.requires_mem = kwargs['requires_mem'] if 'requires_mem' in kwargs else False

    @abc.abstractmethod
    def calc_spike(self):
     
        pass

    def integral(self, inputs):
       
        pass

    def get_thres(self):
        return self.threshold if not self.sigmoid_thres else self.threshold.sigmoid()

    def rearrange2node(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, 'b (c t) w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, 'b (c t) -> t b c', t=self.step)
            else:
                raise NotImplementedError

        elif self.layer_by_layer:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, '(t b) c w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, '(t b) c -> t b c', t=self.step)
            else:
                raise NotImplementedError


        else:
            outputs = inputs

        return outputs

    def rearrange2op(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> b (c t) w h')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> b (c t)')
            else:
                raise NotImplementedError
        elif self.layer_by_layer:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> (t b) c w h')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> (t b) c')
            else:
                raise NotImplementedError

        else:
            outputs = inputs

        return outputs

    def forward(self, inputs):
      

        if self.layer_by_layer or self.groups != 1:
            inputs = self.rearrange2node(inputs)

            outputs = []
            for i in range(self.step):
                
                if self.mem_detach and hasattr(self.mem, 'detach'):
                    self.mem = self.mem.detach()
                    self.spike = self.spike.detach()
                self.integral(inputs[i])
                
                self.calc_spike()
                
                if self.requires_fp is True:
                    self.feature_map.append(self.spike)
                if self.requires_mem is True:
                    self.mem_collect.append(self.mem)
                outputs.append(self.spike)
            outputs = torch.stack(outputs)

            outputs = self.rearrange2op(outputs)
            return outputs
        else:
            if self.mem_detach and hasattr(self.mem, 'detach'):
                self.mem = self.mem.detach()
                self.spike = self.spike.detach()
            self.integral(inputs)
            self.calc_spike()
            if self.requires_fp is True:
                self.feature_map.append(self.spike)
            if self.requires_mem is True:
                self.mem_collect.append(self.mem)   
            return self.spike

    def n_reset(self):
      
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
    def get_n_attr(self, attr):

        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return None

    def set_n_warm_up(self, flag):
       
        self.warm_up = flag

    def set_n_threshold(self, thresh):
       
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)

    def set_n_tau(self, tau):
        
        if hasattr(self, 'tau'):
            self.tau = Parameter(torch.tensor(tau, dtype=torch.float), requires_grad=False)
        else:
            raise NotImplementedError


class BaseMCNode(nn.Module, abc.ABC):
 
    def __init__(self,
                 threshold=1.0,
                 v_reset=0.,
                 comps=[]):
        super().__init__()
        self.threshold = torch.tensor(threshold, requires_grad=False)
        self.v_reset = v_reset
        assert len(comps) != 0
        self.mems = dict()
        for c in comps:
            self.mems[c] = None 
        self.spike = None
        self.warm_up = False

    @abc.abstractmethod
    def calc_spike(self):
        pass
    @abc.abstractmethod
    def integral(self, inputs):
        pass        
    
    def forward(self, inputs: dict):
        '''
        Params:
            inputs dict: Inputs for every compartments of neuron 
        '''
        if self.warm_up:
            return inputs
        else:
            self.integral(**inputs)
            self.calc_spike()
            self.soma_mems_collect.append(self.mems['soma'])
            return self.spike

    def n_reset(self):
        for c in self.mems.keys():
            self.mems[c] = self.v_reset
        self.spike = 0.0

    def get_n_fire_rate(self):
        if self.spike is None:
            return 0.
        return float((self.spike.detach() >= self.threshold).sum()) / float(np.product(self.spike.shape))

    def set_n_warm_up(self, flag):
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)




class MDSNode(BaseMCNode):
#     """
#     多树突脉冲神经元模型
#     :param threshold: 神经元发放脉冲需要达到的阈值
#     :param v_reset: 静息电位
#     :param tau: 胞体膜电位时间常数, 用于控制胞体膜电位衰减
#     :param tau_proximal: 近端树突膜电位时间常数, 用于控制基地树突胞体膜电位衰减
#     :param tau_distal: 远端树突膜电位时间常数, 用于控制远端树突胞体膜电位衰减
#     :param tau_trunk: 主干树突膜电位时间常数, 用于控制远端树突胞体膜电位衰减
#     :param tau_trunk: 
#     :param comps: 神经元不同房室, 例如["distal", "proximal", "trunk",soma"]
#     :param act_fun: 脉冲梯度代理函数
#     """
    def __init__(self, eta=0.01, psi=2.0, C=3.0,alpha=2., nl=1,
                tau=2.0,
                tau_proximal=2.0,
                tau_distal=2.0,
                tau_trunk=2.0,
                v_reset=0.0,
                threshold=1.0,
                comps=[' proximal', 'distal', 'trunk','soma'],
                act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.eta = eta
        self.psi = psi
        self.C = C
        self.tau = tau
        self.alpha = alpha
        self.nl = nl
        self.Vlm = torch.zeros(nl)
        self.Vltheta = torch.zeros(nl)
        self.tau_proximal = tau_proximal
        self.tau_distal = tau_distal
        self.tau_trunk = tau_trunk
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
        if self.mems['soma'] is not None:
            self.mems= torch.tensor(self.mems)
        self.v_reset = torch.tensor(self.v_reset)
        self.start = False
        self.prethreshold = self.threshold

    def dynamic_threshold(self):
        pre_mems = self.soma_mems_collect[-1]
        self.Vlm = torch.mean((pre_mems)) - 0.2 * (torch.max(pre_mems) - torch.min(pre_mems))
        self.Vltheta = torch.mean(self.threshold) - 0.2 * (torch.max(self.threshold) - torch.min(self.threshold))
        Eli = self.eta * (pre_mems- self.Vlm) + self.Vltheta + torch.log(1 + torch.exp((pre_mems - self.Vlm) / self.psi))

        self.Tli = -torch.exp(-torch.mean(self.threshold))
        Tli = self.Tli + torch.exp(-(self.mems['soma']-pre_mems) / self.C)
        return 0.5 *(Eli + Tli)

    def integral(self, proximal_inputs, distal_inputs,trunk_inputs):
        '''
         Params:
            inputs torch.Tensor: Inputs for proximal dendrite  
        '''

        self.mems[' proximal'] =  (self.mems[' proximal'] +  proximal_inputs) / self.tau_proximal
        self.mems['distal'] =  (self.mems['distal'] + distal_inputs) / self.tau_distal
        self.mems['trunk'] =  (self.mems['trunk'] + trunk_inputs) / self.tau_trunk
        self.mems['soma'] = self.mems['soma'] + (self.mems['distal'] + self.mems[' proximal'] + self.mems['trunk']- self.mems['soma']) / self.tau


    def calc_spike(self):
       
        if len( self.soma_mems_collect) > 0:
            self.threshold = self.dynamic_threshold()
            self.threshold = torch.where(self.threshold >= 3, self.prethreshold.cuda(), self.threshold)
            self.threshold = torch.where(self.threshold <= -1, self.prethreshold.cuda(), self.threshold)
            
      
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['proximal'] = self.mems[' proximal'] * (1. - self.spike.detach())
        self.mems['distal'] = self.mems['distal']  * (1. - self.spike.detach())   
        self.mems['trunk'] = self.mems['trunk']  * (1. - self.spike.detach())
        
    def n_reset(self):
        for c in self.mems.keys():
            self.mems[c] = self.v_reset
        
        self.spike = 0.
        self.threshold = self.prethreshold
        self.soma_mems_collect = []
        self.proximal_mems_collect = []
        self.distal_mems_collect = []
        self.trunk_mems_collect = []











class IFNode(BaseNode):

    def __init__(self, threshold=.5, act_fun=AtanGrad, *args, **kwargs):
        """
        :param threshold:
        :param act_fun:
        :param args:
        :param kwargs:
        """
        super().__init__(threshold, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)

    def integral(self, inputs):
        self.mem = self.mem + inputs * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())

  
class LIFNode(BaseNode):
        def __init__(self, eta=0.01, psi=2.0, C=3.0, alpha=2., nl=1,tau=2., act_fun=QGateGrad, threshold=0.5,*args, **kwargs):
            super().__init__(threshold=threshold,*args, **kwargs)
            self.tau = tau
            self.eta = eta
            self.psi = psi
            self.C = C
            self.alpha = alpha
            self.nl = nl
            self.Vlm = torch.zeros(nl)
            self.Vltheta = torch.zeros(nl)
            if isinstance(act_fun, str):
                act_fun = eval(act_fun)
            self.act_fun = act_fun(alpha=2., requires_grad=False)
            self.mem = torch.tensor(self.mem)
            self.prethreshold = self.threshold
            self.start = False
            self.v_reset = torch.tensor(self.v_reset)
        
        def dynamic_threshold(self):
            pre_mem = self.mem_collect[-1]
            self.Vlm = torch.mean((pre_mem)) - 0.2 * (torch.max(pre_mem) - torch.min(pre_mem))
            self.Vltheta = torch.mean(self.threshold) - 0.2 * (torch.max(self.threshold) - torch.min(self.threshold))
            Eli = self.eta * (pre_mem - self.Vlm) + self.Vltheta + torch.log(1 + torch.exp((pre_mem - self.Vlm) / self.psi))

            self.Tli = -torch.exp(-torch.mean(self.threshold))
            Tli = self.Tli + torch.exp(-(self.mem-pre_mem) / self.C)
            return 0.5 *(Eli + Tli)
            
        def integral(self, inputs):
            self.mem = self.mem + (inputs - self.mem) / self.tau
            

        def calc_spike(self):
         
            if len(self.mem_collect) > 0:
                self.threshold = self.dynamic_threshold()
                self.threshold = torch.where(self.threshold >= 3, self.prethreshold.cuda(), self.threshold)
                self.threshold = torch.where(self.threshold <= -1, self.prethreshold.cuda(), self.threshold)
        
            self.spike = self.act_fun(self.mem - self.threshold)
            self.mem = self.mem * (1 - self.spike.detach())

        def n_reset(self):
            self.mem = self.v_reset
            self.spike = 0.
            self.threshold = self.prethreshold
            self.feature_map = []
            self.mem_collect = []









class PLIFNode(BaseNode):


    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        init_w = -math.log(tau - 1.)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=True)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def integral(self, inputs):
        self.mem = self.mem + ((inputs - self.mem) * self.w.sigmoid()) * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())



class SimHHNode(BaseNode):

    def __init__(self, threshold=50., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        '''
        I = Cm dV/dt + g_k*n^4*(V_m-V_k) + g_Na*m^3*h*(V_m-V_Na) + g_l*(V_m - V_L)
        '''
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.g_Na, self.g_K, self.g_l = torch.tensor(120.), torch.tensor(120), torch.tensor(0.3)  # k 36
        self.V_Na, self.V_K, self.V_l = torch.tensor(120.), torch.tensor(-120.), torch.tensor(10.6)  # k -12
        self.m, self.n, self.h = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        self.mem = 0
        self.dt = 0.01

    def integral(self, inputs):
        self.I_Na = torch.pow(self.m, 3) * self.g_Na * self.h * (self.mem - self.V_Na)
        self.I_K = torch.pow(self.n, 4) * self.g_K * (self.mem - self.V_K)
        self.I_L = self.g_l * (self.mem - self.V_l)
        self.mem = self.mem + self.dt * (inputs - self.I_Na - self.I_K - self.I_L) / 0.02
        # non Na
        # self.mem = self.mem + 0.01 * (inputs -  self.I_K - self.I_L) / 0.02  #decayed
        # NON k
        # self.mem = self.mem + 0.01 * (inputs - self.I_Na - self.I_L) / 0.02  #increase

        self.alpha_n = 0.01 * (self.mem + 10.0) / (1 - torch.exp(-(self.mem + 10.0) / 10))
        self.beta_n = 0.125 * torch.exp(-(self.mem) / 80)

        self.alpha_m = 0.1 * (self.mem + 25) / (1 - torch.exp(-(self.mem + 25) / 10))
        self.beta_m = 4 * torch.exp(-(self.mem) / 18)

        self.alpha_h = 0.07 * torch.exp(-(self.mem) / 20)
        self.beta_h = 1 / (1 + torch.exp(-(self.mem + 30) / 10))

        self.n = self.n + self.dt * (self.alpha_n * (1 - self.n) - self.beta_n * self.n)
        self.m = self.m + self.dt * (self.alpha_m * (1 - self.m) - self.beta_m * self.m)
        self.h = self.h + self.dt * (self.alpha_h * (1 - self.h) - self.beta_h * self.h)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())

    def forward(self, inputs):
        self.integral(inputs)
        self.calc_spike()
        return self.spike

    def n_reset(self):
        self.mem = 0.
        self.spike = 0.
        self.m, self.n, self.h = torch.tensor(0), torch.tensor(0), torch.tensor(0)

    def requires_activation(self):
        return False







