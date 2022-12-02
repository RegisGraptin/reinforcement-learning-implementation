

import torch
from DQN import BasicDQN
from utils.ReplayMemory import ReplayMemory, Transition
import random
from itertools import count
import gym

import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




env = gym.make('CartPole-v1')




memory = ReplayMemory(2000)

input_size = 4
ENV_ACTION_SPACE = 2

BATCH_SIZE = 128

policy_net = BasicDQN(input_size, ENV_ACTION_SPACE).to(device)
target_net = BasicDQN(input_size, ENV_ACTION_SPACE).to(device)

policy_net.train()
target_net.eval()

EPSILON = 0.95
TARGET_UPDATE = 100     # Update the network 
GAMMA = 0.99

loss = nn.MSELoss()
optimizer = optim.Adam(policy_net.parameters(), 5e-4)



def learn():
        
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    batch_state      = torch.stack(batch.state)
    batch_action     = torch.stack(batch.action)
    batch_reward     = torch.stack(batch.reward)
    batch_next_state = torch.stack(batch.next_state)
    batch_done       = torch.stack(batch.done)
    

    # print(batch_done)
    # print("---")
    # print((~batch_done.view(BATCH_SIZE, 1)).float())
    


    q_eval = policy_net(batch_state).gather(1, batch_action)
    q_next = target_net(batch_next_state).detach()
    
    q_target = batch_reward + GAMMA * (~batch_done.view(BATCH_SIZE, 1)).float() * q_next.max(1)[0].view(BATCH_SIZE, 1)
    
    
    l = loss(q_eval, q_target)
    
    # print(l)
    # exit()
    
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    
    

def select_action(state):
    if random.random() < EPSILON: 
        action_value = policy_net.forward(state)
        
        action = torch.argmax(action_value, dim=0).item()
        
        return action
    else:
        return random.randrange(ENV_ACTION_SPACE)


def train():
    
    learn_counter = 0
    
    for i_episode in count():
        state, info = env.reset()
        state = torch.from_numpy(state)
        
        running_reward = 0
        
        for t in range(1, 10000):
            
            action = select_action(state)
            
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.from_numpy(next_state)
            
            running_reward += reward
            
            # Transform the reward vector
            action = torch.tensor([action], device=device, dtype=torch.long)
            reward = torch.tensor([reward], device=device)
            done   = torch.tensor([done], device=device)
            
            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)
            state = next_state
            
            
            if len(memory) >= BATCH_SIZE:
                learn()
                learn_counter += 1
                if done:
                    print("episode {}, the reward is {}".format(i_episode, round(running_reward, 3)))
            
            if done or truncated:
                break
            
            if learn_counter % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            
            
            
      
if __name__ == "__main__":
    train()
