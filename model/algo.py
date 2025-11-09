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




import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.path.scheduler import LinearVPScheduler, CondOTScheduler
from flow_matching.path import AffineProbPath, CondOTProbPath, GeodesicProbPath
from flow_matching.utils.manifolds import Sphere, FlatTorus
import pickle

import copy
from .utils import soft_update, hard_update
from .model import QNetwork, ValueNetwork, Policy_flow
# from .model import QNetwork, ValueNetwork, PolicySampler
import time
from torch.optim import Adam
import torch.optim as optim
import numpy as np


CFM_MIN = 1e-3
CFM_MAX = 1

mode = "max-autotune"
compile_model = True

class flowAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.quantile = args.quantile
        self.lamda = args.lamda
        self.noise_level = args.epsilon
        self.action_space = action_space
        self.sample_count = 0

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.device = torch.device(f"cuda:{args.device}" if args.cuda else "cpu")
        self.amp_enabled = args.cuda and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 
        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        self.method = args.method
        manifold = Sphere if args.manifold == "Sphere" else FlatTorus
        if args.path == "Affine":
            self.path = AffineProbPath(LinearVPScheduler())
        elif args.path == "CondOT":
            self.path = CondOTProbPath()
        else:
            self.path = GeodesicProbPath(CondOTScheduler(), manifold())
        # ----------------------  ----------------------
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)  
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.cnt = 0
        self.trigger = 0

        if self.policy_type == "Flow":
            self.policy = Policy_flow(num_inputs, action_space.shape[0], args.hidden_size, args.steps, action_space).to(self.device)
            # self.policy = PolicySampler(num_inputs, action_space.shape[0], args.hidden_size, args.steps, action_space).to(self.device)
            self.policy_optim = optim.Adam(self.policy.parameters(), lr=args.lr)
        else:
            pass

        # ----------------------  ----------------------
        self.critic_buffer = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)

        self.critic_buffer_optim = optim.Adam(self.critic_buffer.parameters(), lr=args.lr)
        hard_update(self.critic_buffer, self.critic)
        self.critic_target_buffer = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target_buffer, self.critic_buffer)
        self.V_critic_buffer = ValueNetwork(num_inputs, args.hidden_size).to(device=self.device)
        self.V_critic_buffer_optim = optim.Adam(self.V_critic_buffer.parameters(), lr=args.lr)
        if compile_model:
            self.critic = torch.compile(self.critic, mode = mode )  
            self.critic_target = torch.compile(self.critic_target, mode = mode)
            self.critic_buffer = torch.compile(self.critic_buffer,mode = mode)
            self.critic_target_buffer = torch.compile(self.critic_target_buffer, mode = mode) 
            self.V_critic_buffer = torch.compile(self.V_critic_buffer, mode = mode)
            # self.policy = torch.compile(self.policy, mode=mode)  
    
    # only use for env step 
    def select_action(self, state, evaluate=False):

        # Noise schedule for exploration: In all tasks, we set the noise to 0.
        if not evaluate:
            self.sample_count += 1
            if self.sample_count % 1e5 == 0:
                self.noise_level = self.noise_level*0.8

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if not evaluate:
            # action, _, _ = self.policy.sample_env(state)
            action, _, _ = self.policy.sample(state, use_ode_solver=True, method=self.method)
            noise = torch.rand_like(action) * 0.01 * self.noise_level
            noise = torch.clamp(noise, -0.25, 0.25)
            action = action + noise
        else:
            with torch.no_grad():
                # _, _, action = self.policy.sample_env(state)
                _, _, action = self.policy.sample(state, use_ode_solver=True, method=self.method)
        
        return action.detach().cpu().numpy()[0].clip(self.action_space.low, self.action_space.high)

    @torch.compile(mode=mode)
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            with torch.no_grad():
                next_state_action, _, _ = self.policy.sample(next_state_batch, use_ode_solver=True, method=self.method)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            next_q_clone = next_q_value.clone()
            qf1, qf2 = self.critic(state_batch, action_batch)
            qf1_loss = F.mse_loss(qf1, next_q_value) 
            qf2_loss = F.mse_loss(qf2, next_q_clone)
            qf_loss = qf1_loss + qf2_loss
        
        self.critic_optim.zero_grad()
        self.scaler.scale(qf_loss).backward()
        self.scaler.step(self.critic_optim)
        self.scaler.update()
        
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            with torch.no_grad():
                target_Vf_pred = self.V_critic_buffer(next_state_batch)
                next_q_value_buffer = reward_batch + mask_batch * self.gamma * target_Vf_pred
            
            qf1_buffer, qf2_buffer = self.critic_buffer(state_batch, action_batch)
            q_buffer = torch.min(qf1_buffer, qf2_buffer)
            qf1_buffer_loss = F.mse_loss(qf1_buffer, next_q_value_buffer)  
            qf2_buffer_loss = F.mse_loss(qf2_buffer, next_q_value_buffer)
            qf_buffer_loss = qf1_buffer_loss + qf2_buffer_loss
        
        self.critic_buffer_optim.zero_grad()
        self.scaler.scale(qf_buffer_loss).backward()
        self.scaler.step(self.critic_buffer_optim)
        self.scaler.update()
        
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            vf_pred = self.V_critic_buffer(state_batch)
            
            with torch.no_grad():
                q_pred_1, q_pred_2 = self.critic_target_buffer(state_batch, action_batch)
                q_pred = torch.min(q_pred_1, q_pred_2)

            vf_err = q_pred - vf_pred
            vf_sign = (vf_err < 0).float()
            vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
            vf_loss = (vf_weight * (vf_err ** 2)).to(torch.float32).mean()
        
        self.V_critic_buffer_optim.zero_grad()
        self.scaler.scale(vf_loss).backward()
        self.scaler.step(self.V_critic_buffer_optim)
        self.scaler.update()

        return q_buffer.detach().clone()
        

    @torch.compile(mode=mode)
    def update_policy(self, state_batch, action_batch, action_0, q_buffer):
        # torch.compiler.cudagraph_mark_step_begin()
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            pi, _, _ = self.policy.sample(state_batch, use_ode_solver=True, method=self.method)
            # explore_ = 0.02 * torch.randn_like(action_batch,device=self.device)
            # explore_ = torch.clamp(explore_, -0.1, 0.1)

            # explore_action = action_batch + explore_
            # with torch.no_grad():
            #     q_buffer1,q_buffer2 = self.critic_buffer(state_batch, explore_action)
            #     q_buffer_ = torch.min(q_buffer1, q_buffer2)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            weights = torch.relu(q_buffer - min_qf_pi.clone()).detach()
            weights = torch.exp(weights- weights.mean())
            weights = self.lamda * weights
            weights= torch.clamp(weights, CFM_MIN, CFM_MAX)
            # velocity_field = action_batch - action_0
            t = torch.rand(action_batch.shape[0], 1).to(self.device)
            # action_t = t * action_batch+ (1. - t) * action_0
            # t = torch.rand(action_batch.shape[0]).to(self.device)
            path_sample = self.path.sample(t=t.squeeze(), x_0=action_0, x_1=action_batch)
            # cfmloss = F.mse_loss(self.policy(state_batch, action_t, t), velocity_field, reduction='mean')            
            cfmloss = F.mse_loss(self.policy(state_batch, path_sample.x_t, t), path_sample.dx_t, reduction='mean')
            cfmloss = weights * cfmloss
            policy_loss = (-min_qf_pi + cfmloss).mean()
        self.policy_optim.zero_grad()
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.policy_optim)
        self.scaler.update()
    


    def update_parameters(self, memory, batch_size, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        action_0 = torch.randn_like(action_batch, device=self.device)
        action_0 = torch.clamp(action_0, -1, 1)

        q_buffer = self.update_critic(
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch
        )
        
        if updates % self.target_update_interval == 0:
            self.update_policy(state_batch, action_batch, action_0, q_buffer)
            with torch.no_grad():
                soft_update(self.critic_target, self.critic, self.tau)
                soft_update(self.critic_target_buffer, self.critic_buffer, self.tau)
        else:
            policy_loss, cfmloss, qf_pi = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device),torch.tensor(0.0, device=self.device)
        

    # Save model parameters
    def save_checkpoint(self, path, i_episode):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    'critic_buffer_state_dict': self.critic_buffer.state_dict(),
                    'critic_target_buffer_state_dict': self.critic_target_buffer.state_dict(),
                    'critic_buffer_optimizer_state_dict': self.critic_buffer_optim.state_dict(),
                    'V_critic_buffer_state_dict': self.V_critic_buffer.state_dict(),
                    'V_critic_buffer_optimizer_state_dict': self.V_critic_buffer_optim.state_dict()
                    },
                    ckpt_path)
    
    # Load model parameters
    def load_checkpoint(self, path, i_episode, evaluate=False):
        # ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        ckpt_path = path + '/' + 'checkpoint/'+'best.torch'
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.critic_buffer.load_state_dict(checkpoint['critic_buffer_state_dict'])
            self.critic_target_buffer.load_state_dict(checkpoint['critic_target_buffer_state_dict'])
            self.critic_buffer_optim.load_state_dict(checkpoint['critic_buffer_optimizer_state_dict'])
            self.V_critic_buffer.load_state_dict(checkpoint['V_critic_buffer_state_dict'])
            self.V_critic_buffer_optim.load_state_dict(checkpoint['V_critic_buffer_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
                self.critic_buffer.eval()
                self.critic_target_buffer.eval()
                self.V_critic_buffer.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
                self.critic_buffer.train()
                self.critic_target_buffer.train()
                self.V_critic_buffer.train()
