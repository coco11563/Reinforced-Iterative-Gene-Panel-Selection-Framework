import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable
import torch.utils.data as Data

from sklearn.metrics import normalized_mutual_info_score

np.random.seed(0)

def Feature_DB(X):
    feature_matrix = []
    for i in range(8):
        feature_matrix = feature_matrix + list(X.astype(np.float64).
                                               describe().iloc[i, :].describe().fillna(0).values)
    return feature_matrix

def _feature_state_generation_des(X):
    feature_matrix = []
    for i in range(8):
        feature_matrix = feature_matrix + list(X.astype(np.float64).
            describe().iloc[i, :].describe().fillna(0).values)
    return feature_matrix

def Feature_GCN(X):
    corr_matrix = X.corr().abs()
    corr_matrix[np.isnan(corr_matrix)] = 0
    corr_matrix_ = corr_matrix - np.eye(len(corr_matrix), k=0)
    sum_vec = corr_matrix_.sum()

    for i in range(len(corr_matrix_)):
        corr_matrix_.iloc[:, i] = corr_matrix_.iloc[:, i] / sum_vec[i]
        corr_matrix_.iloc[i, :] = corr_matrix_.iloc[i, :] / sum_vec[i]
    W = corr_matrix_ + np.eye(len(corr_matrix), k=0)
    Feature = np.mean(np.dot(X.values, W.values), axis=1)

    return Feature


class AutoEncoder(nn.Module):
    def __init__(self, N_feature, N_HIDDEN=4):
        self.N_feature = N_feature
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.N_feature, N_HIDDEN * 4),
            nn.Tanh(),
            nn.Linear(N_HIDDEN * 4, N_HIDDEN * 2),
            nn.Tanh(),
            nn.Linear(N_HIDDEN * 2, N_HIDDEN)
        )
        self.decoder = nn.Sequential(
            nn.Linear(N_HIDDEN, N_HIDDEN * 2),
            nn.Tanh(),
            nn.Linear(N_HIDDEN * 2, N_HIDDEN * 4),
            nn.Tanh(),
            nn.Linear(N_HIDDEN * 4, self.N_feature)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    


def Feature_AE(X, N_HIDDEN=4):
    N_feature = X.shape[1] 
    autoencoder = AutoEncoder(N_feature, N_HIDDEN=N_HIDDEN).cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    loss_func = nn.MSELoss()

    X_tensor = torch.Tensor(X.values).cuda()
    train_loader = Data.DataLoader(dataset=X_tensor, batch_size=104, shuffle=True)
    for epoch in range(10):
        for x in train_loader:
            b_x = Variable(x.view(-1, N_feature)).float()
            encoded, decoded = autoencoder.forward(b_x)
            optimizer.zero_grad()
            loss = loss_func(decoded, b_x)
            loss.backward()
            optimizer.step()
    X_encoded = autoencoder.forward(X_tensor)[0][0].cpu().detach().numpy()
    return X_encoded


def cal_relevance(X, y):
    if len(X.shape) == 1:
        return normalized_mutual_info_score(X, y)
    else:
        N_col = X.shape[1]
        _sum = 0
        for i in range(N_col):
            _sum += normalized_mutual_info_score(X.iloc[:, i], y)
        return _sum / X.shape[1]


def cal_redundancy(X):
    if len(X.shape) == 1:
        return 1
    else:
        N_col = X.shape[1]
        _sum = 0
        for i in range(N_col):
            for j in range(N_col):
                _sum += normalized_mutual_info_score(X.iloc[:, i], X.iloc[:, j])
        return _sum / X.shape[1] ** 2
    

class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)  
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(object):

    def __init__(self, N_STATES, N_ACTIONS,MEMORY_CAPACITY,TARGET_REPLACE_ITER,LR=0.01,GAMMA = 0.9,BATCH_SIZE=32):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.GAMMA=GAMMA
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.BATCH_SIZE=BATCH_SIZE
        self.TARGET_REPLACE_ITER=TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY=MEMORY_CAPACITY
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x,EPSILON,threshold=None):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
            action_value=torch.softmax(action_value,1)

            if threshold==None:
                return 1 if action_value[0][1]>action_value[0][0] else 0
            else:
                return 1 if action_value[0][1]>threshold else 0
        else:
            action = np.random.randint(0, self.N_ACTIONS)

        return action


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.MEMORY_CAPACITY  
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + 1])
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES + 1:self.N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()