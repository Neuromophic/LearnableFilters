import torch

#===============================================================================
#============================ Single Spiking Neuron ============================
#===============================================================================

class SpikingNeuron(torch.nn.Module):
    def __init__(self, beta=None, threshold=None, random_state=True):
        super().__init__()
        
        # initialize beta
        if beta is None:
            beta = torch.rand(1)
        beta_ = torch.log(beta / (1 - beta))
        self.beta_ = torch.nn.Parameter(beta_, requires_grad=True)
        
        # initialize threshold
        if threshold is None:
            threshold = torch.rand(1)
        self.threshold = torch.nn.Parameter(threshold, requires_grad=True)

        # whether to initialize the initial state randomly for simulating unknown previous state
        # this is especially useful for signals split by sliding windows
        self.random_state = random_state

    @property
    def beta(self):
        # keep beta in range (0, 1)
        return torch.sigmoid(self.beta_)
    
    def SpikeGenerator(self, surplus):
        # straight through estimator
        # forward is 0/1 spike
        forward = (surplus >= 0).float()
        # backward is sigmoid relaxation
        backward = torch.sigmoid(surplus)
        return forward.detach() + backward - backward.detach()
    
    def StateUpdate(self, x):
        # update the state of neuron with new input
        return self.beta * self.memory + x
    
    @property
    def fire(self):
        # detect whether the neuron fires
        return self.SpikeGenerator(self.memory - self.threshold)
    
    def StatePostupdate(self):
        # update the state of neuron after firing
        return self.memory - self.fire.detach() * self.threshold.detach()    

    def SingleStepForward(self, x):
        self.memory = self.StateUpdate(x)
        self.spike = self.fire
        self.memory = self.StatePostupdate()
        return self.spike, self.memory
    
    def forward(self, x):
        N_batch, T = x.shape
        # initialize the initial memory to match the batch size
        if self.random_state:
            self.memory = torch.rand(N_batch) * self.threshold.detach()
        else:
            self.memory = torch.zeros(N_batch)
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
    

#===============================================================================
#========================= Spiking Neurons in a Layer ==========================
#===============================================================================

class SpikingLayer(torch.nn.Module):
    def __init__(self, N_neuron, beta=None, threshold=None, random_state=True, spike_only=True):
        super().__init__()
        # create a list of neurons, number of neurons is the number of channels
        self.SNNList = torch.nn.ModuleList()
        for n in range(N_neuron):
            self.SNNList.append(SpikingNeuron(beta, threshold, random_state))
        # whether to output only spikes, False is geneally for visualization
        self.spike_only = spike_only
    
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

#===============================================================================
#==================== Weighted-sum for Temporal Signal =========================
#===============================================================================

class TemporalWeightedSum(torch.nn.Module):
    def __init__(self, N_in, N_out):
        super().__init__()
        # this paameter contrains both weights and bias
        self.weight = torch.nn.Parameter(torch.rand(N_in+1, N_out)/100.)
    
    def forward(self, x):
        # extend the input with 1 for bias
        x_extend = torch.cat([x, torch.ones(x.shape[0], 1, x.shape[2])], dim=1)
        # for each time step, compute the weighted sum
        T = x.shape[2]
        result = []
        for t in range(T):
            mac = torch.matmul(x_extend[:, :, t], self.weight)
            result.append(mac)
        return torch.stack(result, dim=2)
    
#===============================================================================
#======================== Spiking Neural Network ===============================
#===============================================================================

class SpikingNeuralNetwork(torch.nn.Module):
    def __init__(self, topology, beta=None, threshold=None, random_state=True, spike_only=True):
        super().__init__()
        # create snn with weighted-sum and spiking neurons
        self.model = torch.nn.Sequential()
        for i in range(len(topology)-1):
            self.model.add_module('MAC'+str(i), TemporalWeightedSum(topology[i], topology[i+1]))
            self.model.add_module('SNNLayer'+str(i), SpikingLayer(topology[i+1], beta, threshold, random_state))
        # in the complete spiking neural network, the output should be spike only
        self.spike_only = spike_only

    def forward(self, x):
        return self.model(x)
    
    def ResetOutput(self, spike_only, Layer=None):
        self.spike_only = spike_only
        if Layer is None:
            for layer in self.model:
                if hasattr(layer, 'ResetOutput'):
                    layer.ResetOutput(spike_only)
        else:
            for l, layer in enumerate(self.model):
                if l==Layer:
                    try:
                        layer.ResetOutput(spike_only)
                        print(f"Reset output of layer {str(Layer)}.")
                    except:
                        print(f"Layer {str(Layer)} is not a spiking layer, it has no attribute 'ResetOutput'.")


#===============================================================================
#============================= Loss Functin ====================================
#===============================================================================

class SNNLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, output, label):
        num_steps = output.shape[2]
            
        L = torch.tensor(0.)
        
        for step in range(num_steps):
            L += self.loss_fn(output[:,:,step], label)
        return L / num_steps
        
    
