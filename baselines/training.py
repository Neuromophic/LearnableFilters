import torch
import math
import snntorch as snn

def training(nn, loss_fn, optimizer, X_train, y_train, X_valid, y_valid, X_test, y_test):
    early_stop = False
    best_loss = math.inf
    patience = 0
    epoch = 0
    
    while not early_stop:
        y_pred_train = nn(X_train)
        loss_train = loss_fn(y_pred_train, y_train)
        acc_train = (y_pred_train.argmax(dim=1) == y_train).float().mean()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_valid = nn(X_valid)
            loss_valid = loss_fn(y_pred_valid, y_valid)
            acc_valid = (y_pred_valid.argmax(dim=1) == y_valid).float().mean()

            y_pred_test = nn(X_test)
            acc_test = (y_pred_test.argmax(dim=1) == y_test).float().mean()

        if loss_valid < best_loss:
            best_loss = loss_valid
            best_nn = nn
            patience = 0
        else:
            patience += 1
        
        if patience > 500:
            early_stop = True
            break
        
        if (epoch % 100)==0:
            print(f'epoch: {epoch:-8d} | train loss: {loss_train:.5e} | valid loss: {loss_valid:.5e} | train acc: {acc_train:.4f} | valid acc: {acc_valid:.4f} | test acc: {acc_test:.4f} | patience: {patience}')
        
        epoch += 1
        
    if early_stop:
        return best_nn
    else:
        return False
    
def training_snn(nn, loss_fn, optimizer, X_train, y_train, X_valid, y_valid, X_test, y_test):
    early_stop = False
    best_loss = math.inf
    patience = 0
    epoch = 0
    
    while not early_stop:
        y_pred_train = nn(X_train)
        loss_train = loss_fn(y_pred_train, y_train)
        acc_train = (y_pred_train.sum(2).argmax(dim=1) == y_train).float().mean()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_valid = nn(X_valid)
            loss_valid = loss_fn(y_pred_valid, y_valid)
            acc_valid = (y_pred_valid.sum(2).argmax(dim=1) == y_valid).float().mean()

            y_pred_test = nn(X_test)
            acc_test = (y_pred_test.sum(2).argmax(dim=1) == y_test).float().mean()

        if loss_valid < best_loss:
            best_loss = loss_valid
            best_nn = nn
            patience = 0
        else:
            patience += 1
        
        if patience > 500:
            early_stop = True
            break
        
        if (epoch % 10)==0:
            print(f'epoch: {epoch:-8d} | train loss: {loss_train:.5e} | valid loss: {loss_valid:.5e} | train acc: {acc_train:.4f} | valid acc: {acc_valid:.4f} | test acc: {acc_test:.4f} | patience: {patience}')
        
        epoch += 1
        
    if early_stop:
        return best_nn
    else:
        return False
    
class lstm(torch.nn.Module):
    def __init__(self, N_channel, N_class):
        super().__init__()
        self.rnn = torch.nn.LSTM(N_channel, N_class, 1, batch_first=True)
        self.mac = torch.nn.Linear(N_class, N_class)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, X):
        X = X.permute(0,2,1)
        output, (h,c) = self.rnn(X)
        weighted_sum = self.mac(output[:,-1,:])
        result = self.softmax(weighted_sum)
        return result
    
    
class cnn(torch.nn.Module):
    def __init__(self, N_channel, N_length, N_class):
        super().__init__()
        
        stride = max(2,int(N_class/2))
        size = N_class
        padding = int((N_class-1)/2)

        L_in = N_length
        L_outs = []
        while L_in > N_class:
            L_out = int((L_in+2*padding-size) / stride) + 1
            L_outs.append(L_out)
            L_in = L_out
        L_outs.pop()
        
        N_channels = torch.linspace(N_channel, 1, len(L_outs)+1).round().long()

        self.model = torch.nn.Sequential()
        for i in range(len(L_outs)):
            self.model.add_module(f'{i}_conv', torch.nn.Conv1d(N_channels[i], N_channels[i+1], size, stride=stride, padding=padding))
            self.model.add_module(f'{i}_act', torch.nn.PReLU())
        self.model.add_module(f'{i+1}', torch.nn.Flatten())
        self.model.add_module(f'{i+2}', torch.nn.Linear(L_outs[-1],N_class))
        self.model.add_module(f'{i+3}', torch.nn.Softmax(dim=1))

    def forward(self, X):
        return self.model(X)
    


# Define Network
class SNN1(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

        # initialize layers
        self.snnlayer1 = torch.nn.ModuleList()
        for i in range(num_inputs):
            self.snnlayer1.append(snn.Leaky(beta=torch.rand([]), learn_beta=True,
                                            threshold=torch.rand([]), learn_threshold=True, init_hidden=True))
        self.fc1 = torch.nn.Linear(num_inputs, num_hidden)
        self.snnlayer2 = torch.nn.ModuleList()
        for i in range(num_hidden):
            self.snnlayer2.append(snn.Leaky(beta=torch.rand([]), learn_beta=True,
                                            threshold=torch.rand([]), learn_threshold=True, init_hidden=True))
        self.fc2 = torch.nn.Linear(num_hidden, num_outputs)
        self.snnlayer3 = torch.nn.ModuleList()
        for i in range(num_outputs):
            self.snnlayer3.append(snn.Leaky(beta=torch.rand([]), learn_beta=True,
                                            threshold=torch.rand([]), learn_threshold=True, init_hidden=True))
        
    def pass_sigle_sn(self, input, sn):
        num_steps = input.shape[1]
        mem = sn.init_leaky()

        spk_rec = []  # Record the output trace of spikes
        mem_rec = []  # Record the output trace of membrane potential
        
        for step in range(num_steps):
            spk, mem = sn(input[:,step], mem)
            spk_rec.append(spk)
            mem_rec.append(mem)

        return torch.stack(spk_rec).T, torch.stack(mem_rec).T
    
    def pass_layer(self, input, snlist):
        spk_rec = []
        mem_rec = []
        for i in range(len(snlist)):
            spk, mem = self.pass_sigle_sn(input[:,i,:], snlist[i])
            spk_rec.append(spk)
            mem_rec.append(mem)
        return torch.stack(spk_rec, dim=1), torch.stack(mem_rec, dim=1)

    def temproal_mac(self, x, mac):
        num_steps = x.shape[2]
        out = []
        for step in range(num_steps):
            out.append(mac(x[:,:,step]))
        return torch.stack(out, dim=2)

    def forward(self, x):
        num_steps = x.shape[2]
        
        x, mem = self.pass_layer(x, self.snnlayer1)
        x = self.temproal_mac(x, self.fc1)
        x, mem = self.pass_layer(x, self.snnlayer2)
        x = self.temproal_mac(x, self.fc2)
        x, mem = self.pass_layer(x, self.snnlayer3)
        self.spikes = x
        self.mem = mem
        return mem
    

class SNN2(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

        # initialize layers
        self.lif1 = snn.Leaky(beta=torch.rand([]), learn_beta=True,
                              threshold=torch.rand([]), learn_threshold=True, init_hidden=True)
        self.fc1 = torch.nn.Linear(num_inputs, num_hidden)
        self.lif2 = snn.Leaky(beta=torch.rand([]), learn_beta=True,
                              threshold=torch.rand([]), learn_threshold=True, init_hidden=True)
        self.fc2 = torch.nn.Linear(num_hidden, num_outputs)
        self.lif3 = snn.Leaky(beta=torch.rand([]), learn_beta=True,
                              threshold=torch.rand([]), learn_threshold=True, init_hidden=True)

    def forward(self, x):
        num_steps = x.shape[2]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []  # Record the output trace of spikes
        mem_rec = []  # Record the output trace of membrane potential

        for step in range(num_steps):
            spk1, mem1 = self.lif1(x[:,:,step], mem1)
            curl1 = self.fc1(spk1)
            spk2, mem2 = self.lif2(curl1, mem2)
            curl2 = self.fc2(spk2)
            spk3, mem3 = self.lif3(curl2, mem3)
            spk_rec.append(spk3)
            mem_rec.append(mem3)
        self.spikes = torch.stack(spk_rec, dim=2)
        self.mem = torch.stack(mem_rec, dim=2)
        return self.mem
    

class SNN3(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

        self.fc1 = torch.nn.Linear(num_inputs, num_hidden)
        self.lif2 = snn.Leaky(beta=torch.rand([]), learn_beta=True,
                              threshold=torch.rand([]), learn_threshold=True, init_hidden=True)
        self.fc2 = torch.nn.Linear(num_hidden, num_outputs)
        self.lif3 = snn.Leaky(beta=torch.rand([]), learn_beta=True,
                              threshold=torch.rand([]), learn_threshold=True, init_hidden=True)

    def forward(self, x):
        num_steps = x.shape[2]

        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []  # Record the output trace of spikes
        mem_rec = []  # Record the output trace of membrane potential

        for step in range(num_steps):
            curl1 = self.fc1(x[:,:,step])
            spk2, mem2 = self.lif2(curl1, mem2)
            curl2 = self.fc2(spk2)
            spk3, mem3 = self.lif3(curl2, mem3)
            spk_rec.append(spk3)
            mem_rec.append(mem3)
        self.spikes = torch.stack(spk_rec, dim=2)
        self.mem = torch.stack(mem_rec, dim=2)
        return self.mem
    

