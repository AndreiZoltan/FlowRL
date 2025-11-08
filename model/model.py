# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper

# Initialize  weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.v_input = nn.Linear(num_inputs, hidden_dim)
        self.block = create_value_block(hidden_dim)
        self.v_output = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = self.v_input(state)
        x = self.block(x)
        x = self.v_output(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        
        # Q1
        self.Q1_input = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.Q1_block = create_value_block(hidden_dim)
        self.Q1_output = nn.Linear(hidden_dim,1)
        
        # Q2
        self.Q2_input = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.Q2_block = create_value_block(hidden_dim)
        self.Q2_output = nn.Linear(hidden_dim,1)
        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        
        # Q1_value
        x1 = self.Q1_input(x)
        x1 = self.Q1_block(x1)
        q_value_1 = self.Q1_output(x1)
        
        # Q2_value
        x2 = self.Q2_input(x)
        x2 = self.Q2_block(x2)
        q_value_2 = self.Q2_output(x2)
        
        return q_value_1, q_value_2


class Policy_flow(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, steps, action_space=None):
        super(Policy_flow, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs + num_actions + 1, hidden_dim)  # add time embedding, now, time_embedding = time
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.LayerNorm2 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)
        self.steps = steps  # num of steps
        self.device = torch.device(f"cuda:0")
        self.apply(weights_init_)
         
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).cuda()
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).cuda()
            
    def forward(self, state, action_0, time):
        x = torch.cat([state, action_0, time], 1)
        x = self.linear1(x)
        x = self.LayerNorm(x)
        x = F.elu(x)
        x = self.linear2(x)
        x = self.LayerNorm2(x)
        x = F.elu(x)
        x = self.linear3(x)
        return x
    

    def step(self, state, action,  time_start, time_end):
        """
        Integrate the velocity field from time_start to time_end using midpoint method
        """
        velocity_start = self.forward(state, action, time_start)
        intermediate_state = action + velocity_start * (time_end - time_start)/2
        
        velocity_mid = self.forward(state, intermediate_state, time_start + (time_end - time_start)/2)
        action_t = action + velocity_mid * (time_end - time_start)

        return action_t
    

    @torch.compile
    def sample(self, state):
        # sampel an action from the nomarl, mean = 0, std = 1
        time_start = torch.zeros(state.shape[0], 1, device = self.device)
        time_step = 1.0 / self.steps  # Assuming we go from t=0 to t=1 in `steps` steps
        action = torch.normal(0, 1, size=(state.shape[0], self.num_actions),device = self.device)
        action = torch.clamp(action,-1.0,1.0)

        for i in range(self.steps):
            time_end = time_start + time_step
            action = self.step(state, action, time_start, time_end)
            time_start = time_end
            
        # action = torch.clamp(action,-1.0,1.0)
        action = torch.tanh(action)
        action = action * self.action_scale + self.action_bias
        return action,0, action

    @torch.compile    
    def sample_env(self, state):
        # sampel an action from the nomarl, mean = 0, std = 1
        time_start = torch.zeros(state.shape[0], 1, device=self.device)
        time_step = 1.0 / self.steps  # Assuming we go from t=0 to t=1 in `steps` steps
        action = torch.normal(0, 1, size=(state.shape[0], self.num_actions),device=self.device)
        action = torch.clamp(action,-1.0,1.0)
        
        for i in range(self.steps):
            time_end = time_start + time_step
            action = self.step(state, action, time_start, time_end)
            time_start = time_end
        
        #action = torch.clamp(action,-1.0,1.0)
        action = torch.tanh(action)
        action = action * self.action_scale + self.action_bias
        return action,0, action

class policy_model(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(policy_model, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions + 1, hidden_dim)  # add time embedding, now, time_embedding = time
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.LayerNorm2 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)
    
    def forward(self, action, time, state):
        time = time.view(1, 1)
        x = torch.cat([state, action, time], 1)
        x = self.linear1(x)
        x = self.LayerNorm(x)
        x = F.elu(x)
        x = self.linear2(x)
        x = self.LayerNorm2(x)
        x = F.elu(x)
        x = self.linear3(x)
        return x
    

class WrappedModel(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras): # x is for action; t for time
        return self.model(x, t, **extras)
    
# May be to define PolicySampler like PolicySampler(ModelWrapper) etc
class PolicySampler(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, steps, action_space=None):
        super(PolicySampler, self).__init__()
        self.device = torch.device(f"cuda:0")
        self.steps = steps
        self.num_actions = num_actions
        model = policy_model(num_inputs, num_actions, hidden_dim)
        self.wrapped_model = WrappedModel(model)
        self.solver = ODESolver(velocity_model=self.wrapped_model)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).cuda(0)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).cuda(0)

    def forward(self, state, action, time):
        return self.wrapped_model(x=action, t=time, state=state)

    # sample an action from the normal, mean = 0, std = 1
    def sample(self, state):
        #TODO solve time start issue
        time_start = torch.zeros(state.shape[0], 1, device=self.device)
        time_step = 1.0 / self.steps  # Assuming we go from t=0 to t=1 in `steps` steps
        action = torch.normal(0, 1, size=(state.shape[0], self.num_actions),device=self.device)
        action = torch.clamp(action,-1.0,1.0)
        print(f"{action.shape=}, {state.shape=}")
        action = self.solver.sample(
            x_init=action,
            step_size=time_step,
            state=state
        )
        assert isinstance(action, torch.Tensor)
        #action = torch.clamp(action,-1.0,1.0)
        action = torch.tanh(action)
        action = action * self.action_scale + self.action_bias
        return action,0, action


def create_value_block(hidden_dim):
    return nn.Sequential(
    nn.LayerNorm(hidden_dim),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),  
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    )
