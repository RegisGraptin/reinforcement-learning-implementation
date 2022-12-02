
import gym
import torch

from itertools import count

from DQN import DQN
from utils.ReplayMemory import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Environment:
    
    def __init__(self) -> None:
        
        self.memory = ReplayMemory(2000)
        
        self.env = gym.make('CartPole-v1')
        
        self.input_size       = 4
        self.env_action_space = 2
        
        self.DQN = DQN(self.input_size, self.env_action_space)
        
    def train(self):

        learn_counter = 0
        
        for i_episode in count():
            state, info = self.env.reset()
            state = torch.from_numpy(state)
            
            running_reward = 0
            
            for t in range(1, 10000):
                
                action = self.DQN.select_action(state)
                
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = torch.from_numpy(next_state)
                
                running_reward += reward
                
                # Transform the reward vector
                action = torch.tensor([action], device=device, dtype=torch.long)
                reward = torch.tensor([reward], device=device)
                done   = torch.tensor([done], device=device)
                
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward, done)
                state = next_state
                
                
                if len(self.memory) >= self.DQN.BATCH_SIZE:
                    transitions = self.memory.sample(self.DQN.BATCH_SIZE)
                    self.DQN.learn(transitions)
                    learn_counter += 1
                    if done:
                        print("episode {}, the reward is {}".format(i_episode, round(running_reward, 3)))
                
                if done or truncated:
                    break
                
                if learn_counter % self.DQN.TARGET_UPDATE == 0:
                    self.DQN.target_net.load_state_dict(self.DQN.policy_net.state_dict())

