import torch 
import torch.nn as nn
import torch.nn.functional as F

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
        
        # x = F.softmax(self.linear4(x), dim=0)
        # # x = self.linear4(x)
        
        
        # # x = F.relu(self.linear3(x))
        # # x = x.view(x.size(0), -1)
        
        # # tensor( [ [0.4527],[0.5473]], device='cuda:0')
        
        # return x
        
        