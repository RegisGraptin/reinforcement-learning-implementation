import random

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.ReplayMemory import ReplayMemory, Transition


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicDQN(nn.Module):
    
    def __init__(self, inputs_size: int, outputs_size: int) -> None:
        super(BasicDQN, self).__init__()
        
        # Define the size of our network
        self.inputs_size  = inputs_size
        self.outputs_size = outputs_size
        
        self.linear1 = nn.Linear(self.inputs_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, self.outputs_size)
    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return x

class DQN:
    
    def __init__(self, input_size: int, env_action_size: int) -> None:

        self.input_size      = input_size
        self.env_action_size = env_action_size
                
        self.policy_net = BasicDQN(input_size, env_action_size).to(device)
        self.target_net = BasicDQN(input_size, env_action_size).to(device)

        self.policy_net.train()
        self.target_net.eval()

        self.EPSILON = 0.95
        self.GAMMA = 0.99
        self.TARGET_UPDATE = 100
        
        self.BATCH_SIZE = 128
        
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), 5e-4)


    def select_action(self, state):
        if random.random() < self.EPSILON: 
            action_value = self.policy_net.forward(state)
            
            action = torch.argmax(action_value, dim=0).item()
            
            return action
        else:
            return random.randrange(self.env_action_size)
        
    def learn(self, transitions):
        
        batch = Transition(*zip(*transitions))
        
        batch_state      = torch.stack(batch.state)
        batch_action     = torch.stack(batch.action)
        batch_reward     = torch.stack(batch.reward)
        batch_next_state = torch.stack(batch.next_state)
        batch_done       = torch.stack(batch.done)
        

        q_eval = self.policy_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        
        q_target = batch_reward + self.GAMMA * (~batch_done.view(self.BATCH_SIZE, 1)).float() * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        
        
        l = self.loss(q_eval, q_target)
        
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
        