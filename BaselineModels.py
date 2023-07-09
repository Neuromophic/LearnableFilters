import numpy as np
import torch
import warnings

# ================================================================================================================================================
# =====================================================  Learnable Negative Weight Circuit  ======================================================
# ================================================================================================================================================


class InvRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # R1n, k1, R3n, k2, R5n, Wn, k3
        # be careful, k1, k2, k3 are not normalized
        self.rt_ = torch.nn.Parameter(torch.tensor(
            [args.NEG_R1n, args.NEG_k1, args.NEG_R3n, args.NEG_k2, args.NEG_R5n, args.NEG_Wn, args.NEG_Ln]), requires_grad=True)
        # model
        package = torch.load('./utils/neg_model_package')
        self.eta_estimator = package['eta_estimator'].to(self.args.DEVICE)
        self.eta_estimator.train(False)
        for name, param in self.eta_estimator.named_parameters():
            param.requires_grad = False
        self.X_max = package['X_max'].to(self.args.DEVICE)
        self.X_min = package['X_min'].to(self.args.DEVICE)
        self.Y_max = package['Y_max'].to(self.args.DEVICE)
        self.Y_min = package['Y_min'].to(self.args.DEVICE)
        # load power model
        package = torch.load('./utils/neg_power_model_package')
        self.power_estimator = package['power_estimator'].to(self.args.DEVICE)
        for name, param in self.power_estimator.named_parameters():
            param.requires_grad = False
        self.power_estimator.train(False)
        self.pow_X_max = package['X_max'].to(self.args.DEVICE)
        self.pow_X_min = package['X_min'].to(self.args.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.args.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.args.DEVICE)
    
    @property
    def RT_(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        # calculate normalized (only R1n, R3n, R5n, Wn, Ln)
        RTn = torch.zeros([10]).to(self.args.DEVICE)
        RTn[0] = rt_temp[0]    # R1n
        RTn[2] = rt_temp[2]    # R3n
        RTn[4] = rt_temp[4]    # R5n
        RTn[5] = rt_temp[5]    # Wn
        RTn[6] = rt_temp[6]    # Ln
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        # calculate R2, R4
        R2 = RT[0] * rt_temp[1]  # R2 = R1 * k1
        R4 = RT[2] * rt_temp[3]  # R4 = R3 * k2
        # stack new variable: R1, R2, R3, R4, R5, W, L
        RT_full = torch.stack([RT[0], R2, RT[2], R4, RT[4], RT[5], RT[6]])
        return RT_full

    @property
    def RT(self):
        # keep each component value in feasible range
        RT_full = torch.zeros([10]).to(self.args.DEVICE)
        RT_full[:7] = self.RT_.clone()
        RT_full[RT_full > self.X_max] = self.X_max[RT_full > self.X_max]    # clip
        RT_full[RT_full < self.X_min] = self.X_min[RT_full < self.X_min]    # clip
        return RT_full[:7].detach() + self.RT_ - self.RT_.detach()

    @property
    def RT_extend(self):
        # extend RT to 10 variables with k1 k2 and k3
        R1 = self.RT[0]
        R2 = self.RT[1]
        R3 = self.RT[2]
        R4 = self.RT[3]
        R5 = self.RT[4]
        W = self.RT[5]
        L = self.RT[6]
        k1 = R2 / R1
        k2 = R4 / R3
        k3 = L / W
        return torch.hstack([R1, R2, R3, R4, R5, W, L, k1, k2, k3])

    @property
    def RTn_extend(self):
        # normalize RT_extend
        return (self.RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power

    def forward(self, z):
        return - (self.eta[0] + self.eta[1] * torch.tanh((z - self.eta[2]) * self.eta[3]))


# ================================================================================================================================================
# ========================================================  Learnable Activation Circuit  ========================================================
# ================================================================================================================================================

class TanhRT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # R1n, R2n, W1n, L1n, W2n, L2n
        self.rt_ = torch.nn.Parameter(
            torch.tensor([args.ACT_R1n, args.ACT_R2n, args.ACT_W1n, args.ACT_L1n, args.ACT_W2n, args.ACT_L2n]), requires_grad=True)

        # model
        package = torch.load('./utils/act_model_package')
        self.eta_estimator = package['eta_estimator'].to(self.args.DEVICE)
        self.eta_estimator.train(False)
        for n, p in self.eta_estimator.named_parameters():
            p.requires_grad = False
        self.X_max = package['X_max'].to(self.args.DEVICE)
        self.X_min = package['X_min'].to(self.args.DEVICE)
        self.Y_max = package['Y_max'].to(self.args.DEVICE)
        self.Y_min = package['Y_min'].to(self.args.DEVICE)
        # load power model
        package = torch.load('./utils/act_power_model_package')
        self.power_estimator = package['power_estimator'].to(self.args.DEVICE)
        self.power_estimator.train(False)
        for n, p in self.power_estimator.named_parameters():
            p.requires_grad = False
        self.pow_X_max = package['X_max'].to(self.args.DEVICE)
        self.pow_X_min = package['X_min'].to(self.args.DEVICE)
        self.pow_Y_max = package['Y_max'].to(self.args.DEVICE)
        self.pow_Y_min = package['Y_min'].to(self.args.DEVICE)

    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        # denormalization
        RTn = torch.zeros([9]).to(self.args.DEVICE)
        RTn[0] = rt_temp[0]    # R1n
        RTn[1] = rt_temp[1]    # R2n
        RTn[2] = rt_temp[2]    # W1n
        RTn[3] = rt_temp[3]    # L1n
        RTn[4] = rt_temp[4]    # W2n
        RTn[5] = rt_temp[5]    # L2n
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT[:6]

    @property
    def RT_extend(self):
        # extend RT to 9 variables with k1 k2 and k3
        R1 = self.RT[0]
        R2 = self.RT[1]
        W1 = self.RT[2]
        L1 = self.RT[3]
        W2 = self.RT[4]
        L2 = self.RT[5]
        k1 = R2 / R1
        k2 = L1 / W1
        k3 = L2 / W2
        return torch.hstack([R1, R2, W1, L1, W2, L2, k1, k2, k3])

    @property
    def RTn_extend(self):
        # normalize RT_extend
        return (self.RT_extend - self.X_min) / (self.X_max - self.X_min)

    @property
    def eta(self):
        # calculate eta
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta

    @property
    def power(self):
        # calculate power
        power_n = self.power_estimator(self.RTn_extend)
        power = power_n * (self.pow_Y_max - self.pow_Y_min) + self.pow_Y_min
        return power.flatten()

    def forward(self, z):
        return self.eta[0] + self.eta[1] * torch.tanh((z - self.eta[2]) * self.eta[3])


# ================================================================================================================================================
# ===============================================================  Printed Layer  ================================================================
# ================================================================================================================================================

class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, args, ACT, INV):
        super().__init__()
        self.args = args
        # define nonlinear circuits
        self.INV = INV
        self.ACT = ACT
        # initialize conductances for weights
        theta = torch.rand([n_in + 2, n_out])/100. + args.gmin
        theta[-1, :] = theta[-1, :] + args.gmax
        theta[-2, :] = self.ACT.eta[2].detach().item() / \
            (1.-self.ACT.eta[2].detach().item()) * \
            (torch.sum(theta[:-2, :], axis=0)+theta[-1, :])
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

    @property
    def device(self):
        return self.args.DEVICE

    @property
    def theta(self):
        self.theta_.data.clamp_(-self.args.gmax, self.args.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < self.args.gmin] = 0.
        return theta_temp.detach() + self.theta_ - self.theta_.detach()

    @property
    def W(self):
        return self.theta.abs() / torch.sum(self.theta.abs(), axis=0, keepdim=True)

    def MAC(self, a):
        # 0 and positive thetas are corresponding to no negative weight circuit
        positive = self.theta.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive
        a_extend = torch.cat([a,
                              torch.ones([a.shape[0], 1]).to(self.device),
                              torch.zeros([a.shape[0], 1]).to(self.device)], dim=1)
        a_neg = self.INV(a_extend)
        a_neg[:, -1] = 0.
        z = torch.matmul(a_extend, self.W * positive) + \
            torch.matmul(a_neg, self.W * negative)
        return z

    def forward(self, a_previous):
        z_new = self.MAC(a_previous)
        a_new = self.ACT(z_new)
        return a_new

    def UpdateArgs(self, args):
        self.args = args
        self.INV.args = args
        self.ACT.args = args


# ================================================================================================================================================
# ==============================================================  Printed Circuit  ===============================================================
# ================================================================================================================================================

class pNN(torch.nn.Module):
    def __init__(self, topology, args):
        super().__init__()

        self.args = args

        # define nonlinear circuits
        self.act = TanhRT(args)
        self.inv = InvRT(args)

        self.model = torch.nn.Sequential()
        for i in range(len(topology)-1):
            self.model.add_module(
                f'{i}-th pLayer', pLayer(topology[i], topology[i+1], args, self.act, self.inv))

    def forward(self, X):
        return self.model(X)

    @property
    def device(self):
        return self.args.DEVICE

    def GetParam(self):
        weights = [p for name, p in self.named_parameters() if name.endswith('.theta_')]
        nonlinear = [p for name, p in self.named_parameters() if name.endswith('.rt_')]
        if self.args.lnc:
            return weights + nonlinear
        else:
            return weights

    def UpdateArgs(self, args):
        self.args = args
        for m in self.model:
            m.UpdateArgs(self.args)

    
# ================================================================================================================================================
# ============================================================= Recurrent Neural Network =========================================================
# ================================================================================================================================================

class RNN(torch.nn.Module):
    def __init__(self, n_feature, n_class, n_layer):
        super().__init__()
        self.model = torch.nn.RNN(n_feature, n_class, n_layer, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        output, _ = self.model(x)
        output = output.permute(0, 2, 1)
        output = output[:, :, -1]
        return output
    
    def GetParam(self):
        return self.model.parameters()
    

#===============================================================================
#============================ Single Spiking Neuron ============================
#===============================================================================
    
class SpikingNeuron(torch.nn.Module):
    def __init__(self, args, random_state=True):
        super().__init__()
        self.args = args
        self.beta = torch.nn.Parameter(torch.randn([]), requires_grad=True)
        self.threshold = torch.nn.Parameter(torch.randn([]), requires_grad=True)

        # whether to initialize the initial state randomly for simulating unknown previous state
        # this is especially useful for signals split by sliding windows
        self.random_state = random_state

    @property
    def feasible_beta(self):
        return torch.sigmoid(self.beta)
    
    @property
    def feasible_threshold(self):
        return torch.nn.functional.softplus(self.threshold)
    
    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    def SpikeGenerator(self, surplus):
        # straight through estimator
        # forward is 0/1 spike
        forward = (surplus >= 0).float()
        # backward is sigmoid relaxation
        backward = torch.sigmoid(surplus * 5.)
        return forward.detach() + backward - backward.detach()
    
    def StateUpdate(self, x):
        # update the state of neuron with new input
        return self.feasible_beta * self.memory + (1. - self.feasible_beta) * x
    
    @property
    def fire(self):
        # detect whether the neuron fires
        return self.SpikeGenerator(self.memory - self.threshold)
    
    def StatePostupdate(self):
        # update the state of neuron after firing
        return self.memory - self.fire.detach() * self.threshold

    def SingleStepForward(self, x):
        self.memory = self.StateUpdate(x)
        self.spike = self.fire
        self.memory = self.StatePostupdate()
        return self.spike, self.memory
    
    def forward(self, x):
        N_batch, T = x.shape
        # initialize the initial memory to match the batch size
        if self.random_state:
            self.memory = torch.rand(N_batch).to(self.DEVICE) * self.threshold.detach()
        else:
            self.memory = torch.zeros(N_batch).to(self.DEVICE)
        # forward
        spikes = []
        memories = [self.memory] # add initial state
        for t in range(T):
            spike, memory = self.SingleStepForward(x[:,t])
            spikes.append(spike)
            memories.append(memory)
        memories.pop() # remove the last one to keep the same length as spikes
        # output
        return torch.stack(spikes, dim=1), torch.stack(memories, dim=1)
    
    def UpdateArgs(self, args):
        self.args = args    

#===============================================================================
#========================= Spiking Neurons in a Layer ==========================
#===============================================================================

class SpikingLayer(torch.nn.Module):
    def __init__(self, args, N_neuron, random_state=True, spike_only=True):
        super().__init__()
        self.args = args
        
        # create a list of neurons, number of neurons is the number of channels
        self.SNNList = torch.nn.ModuleList()
        for n in range(N_neuron):
            self.SNNList.append(SpikingNeuron(args, random_state))
        # whether to output only spikes, False is geneally for visualization
        self.spike_only = spike_only
        
    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    def forward(self, x):
        spikes = []
        memories = []
        # for each channel, run the corresponding spiking neuron
        for c in range(x.shape[1]):
            spike, memory = self.SNNList[c](x[:,c,:])
            spikes.append(spike)
            memories.append(memory)
        if self.spike_only:
            return torch.stack(spikes, dim=1)
        else:
            return torch.stack(spikes, dim=1), torch.stack(memories, dim=1)
    
    def ResetOutput(self, spike_only):
        # reset the output mode
        self.spike_only = spike_only
        
    def UpdateArgs(self, args):
        self.args = args

#===============================================================================
#==================== Weighted-sum for Temporal Signal =========================
#===============================================================================

class TemporalWeightedSum(torch.nn.Module):
    def __init__(self, args, N_in, N_out):
        super().__init__()
        self.args = args
        # this paameter contrains both weights and bias
        self.weight = torch.nn.Parameter(torch.randn(N_in+1, N_out)/10.)
    
    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    def forward(self, x):
        # extend the input with 1 for bias
        x_extend = torch.cat([x, torch.ones(x.shape[0], 1, x.shape[2]).to(self.DEVICE)], dim=1)
        # for each time step, compute the weighted sum
        T = x.shape[2]
        result = []
        for t in range(T):
            mac = torch.matmul(x_extend[:, :, t], self.weight)
            result.append(mac)
        return torch.stack(result, dim=2)
    
    def UpdateArgs(self, args):
        self.args = args
    
#===============================================================================
#======================== Spiking Neural Network ===============================
#===============================================================================

class SpikingNeuralNetwork(torch.nn.Module):
    def __init__(self, args, topology, beta=None, threshold=None, random_state=True):
        super().__init__()
        self.args = args
        
        # create snn with weighted-sum and spiking neurons
        self.model = torch.nn.Sequential()
        for i in range(len(topology)-1):
            self.model.add_module('MAC'+str(i), TemporalWeightedSum(args, topology[i], topology[i+1]))
            self.model.add_module('SNNLayer'+str(i), SpikingLayer(args, topology[i+1], random_state))
        self.InitOutput()
    
    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    def forward(self, x):
        return self.model(x)

    def Power(self):
        # this function is just used to match the evaluation
        # the power of snn in-silico is acutally not considered
        return torch.zeros(1).to(self.DEVICE)

    def InitOutput(self):
        for layer in self.model:
            if hasattr(layer, 'ResetOutput'):
                layer.ResetOutput(True)
        self.model[-1].ResetOutput(False)

    def ResetOutput(self, Layer, spike_only=False):
        self.InitOutput()
        for l, layer in enumerate(self.model):
            if l==Layer:
                try:
                    layer.ResetOutput(spike_only)
                    print(f"The memory of layer {str(Layer)} will be output.")
                except:
                    print(f"Layer {str(Layer)} is not a spiking layer, it has no attribute 'ResetOutput'.")
    
    def UpdateArgs(self, args):
        self.args = args
        self.beta.args = args
        self.threshold.args = args
        for layer in self.model:
            if hasattr(layer, 'UpdateArgs'):
                layer.UpdateArgs(args)
                
    def GetParam(self):
        weights = [p for name, p in self.named_parameters() if name.endswith('weight')]
        nonlinear = [p for name, p in self.named_parameters() if name.endswith('beta') or name.endswith('threshold')]
        # if self.args.lnc:
        #     return weights + nonlinear
        # else:
        #     return weights
        return weights + nonlinear
    

#===============================================================================
#============================= Loss Functin ====================================
#===============================================================================

class SNNLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, model, input, label):
        _, mem = model(input)
        L = []       
        for step in range(mem.shape[2]):
            L.append(self.loss_fn(mem[:,:,step], label))
        return torch.stack(L).mean()
        
    
class LossFN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def standard(self, prediction, label):
        label = label.reshape(-1, 1)
        fy = prediction.gather(1, label).reshape(-1, 1)
        fny = prediction.clone()
        fny = fny.scatter_(1, label, -10 ** 10)
        fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
        l = torch.max(self.args.m + self.args.T - fy, torch.tensor(0)
                      ) + torch.max(self.args.m + fnym, torch.tensor(0))
        L = torch.mean(l)
        return L
    
    def celoss(self, prediction, label):
        fn = torch.nn.CrossEntropyLoss()
        return fn(prediction, label)

    def temporal(self, prediction, label):
        L = []
        T = prediction.shape[2]
        for t in range(T):
            L.append(self.celoss(prediction[:,:,t], label))
        return torch.stack(L).mean()
        

    def forward(self, nn, x, label):
        if self.args.loss == 'pnnloss':
            return self.standard(nn(x), label)
        elif self.args.loss == 'celoss':
            return self.celoss(nn(x), label)
        elif self.args.loss == 'temporal':
            L = []
            T = x.shape[2]
            for t in range(T):
                L.append(self.celoss(nn(x[:,:,t]), label))
            return torch.stack(L).mean()