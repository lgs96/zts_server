from operator import truediv
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import easydict
from torch.optim import lr_scheduler
from datetime import datetime

from agents.common.utils import *
from agents.common.buffers import *
from agents.common.networks import *
from agents.common.dueling import *
from agents.dqn import *

class BranchingQNetwork(nn.Module):

    def __init__(self, obs: int, ac_dim: int, n: int): 

        super().__init__()

        self.ac_dim = ac_dim
        self.n = n 

        self.model = nn.Sequential(nn.Linear(obs, 8), 
                                   nn.ReLU(),
                                   nn.Linear(8, 4), 
                                   nn.ReLU())

        self.value_head = nn.Linear(4, 1)
        
        self.adv_heads = nn.ModuleList([nn.Linear(4, n) for i in range(ac_dim)])

    def forward(self, x): 
        #print('Forward ', x,  x.shape) 

        out = self.model(x)
        #print('Forward1 ',  x.shape) 
        value = self.value_head(out)
        #print('Forward2 ',  x.shape) 
        advs = torch.stack([l(out) for l in self.adv_heads], dim = 1)
        #print('Forward3 ', x,  x.shape, advs.shape, advs) 

        test =  advs.mean(2, keepdim = True)
        #print('Shape: ', advs.shape, test.shape, value.shape, value.unsqueeze(2).shape)

        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim = True)
        #print('Shape 2: ', advs.shape, test.shape, value.shape, value.unsqueeze(2).shape)

        return q_val

class Agent(object):
   """An implementation of the Deep Q-Network (DQN), Double DQN agents."""

   def __init__(self,
                #env,
                args,
                device,
                obs_dim,
                act_dim,
                act_num,
                steps=0,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=0.985,
                buffer_size=int(500),
                batch_size=64,
                target_update_step=50,
                eval_mode=False,
                q_losses=list(),
                logger=dict(),
   ):

      #self.env = env
      self.args = args
      self.device = device
      self.obs_dim = obs_dim
      self.act_dim = act_dim
      self.act_num = act_num
      self.steps = steps
      self.gamma = gamma
      self.epsilon = epsilon
      self.epsilon_decay = epsilon_decay
      self.buffer_size = buffer_size
      self.batch_size = batch_size
      self.target_update_step = target_update_step
      self.eval_mode = eval_mode
      self.q_losses = q_losses
      self.logger = logger

      self.save_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

      ## reward 
      self.target_fps = 30
      self.max_res = 38.4
      self.beta1 = 0.7
      self.beta2 = 0.3
      self.current_buffer_index = 0
      self.max_current_buffer_size = 5
      self.current_buffer = [None] * self.max_current_buffer_size
      self.end = False

      # Main network
      self.qf = BranchingQNetwork(self.obs_dim, self.act_dim, self.act_num).to(self.device)
      if args.load_saved == True:
         print('Load saved')
         self.qf.load_state_dict(torch.load(self.args.load_path))
      # Target network
      self.qf_target = BranchingQNetwork(self.obs_dim, self.act_dim, self.act_num).to(self.device)
      
      # Initialize target parameters to match main parameters
      hard_target_update(self.qf, self.qf_target)

      # Create an optimizer
      self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=0.05)

      # Experience buffer
      self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size, self.device)

   def select_action(self, obs, data):
      """Select an action from the set of available actions."""
      # Decaying epsilon
      self.epsilon *= self.epsilon_decay
      self.epsilon = max(self.epsilon, 0.02)
      action = np.array([0,0])

      if np.random.rand() <=  self.epsilon:
         # Choose a random action with probability epsilon
         action =  np.random.randint(self.act_num, size = self.act_dim)
         #print('Random action: ', action)
      else:
         # Choose the action with highest Q-value at the current state
         q_value = self.qf(obs)
         q_argmax = torch.argmax(q_value.squeeze(0), dim = 1)
         action =  q_argmax.detach().cpu().numpy()
         #print('RL action: ', action)

      # cool down action
      if data['modem_temp']/1000 >= 51:
         action = np.array([0, 0, 0])

      #print('Action: ', action, data['modem_temp']/1000)

      return action

   def train_model(self):
      batch = self.replay_buffer.sample(self.batch_size)
      obs1 = batch['obs1']
      obs2 = batch['obs2']
      acts = batch['acts']
      rews = batch['rews']
      done = batch['done']

      if 0: # Check shape of experiences
         print("obs1", obs1.shape, obs1)
         print("obs2", obs2.shape, self.qf(obs2).shape, torch.argmax(self.qf(obs2), dim = 2))
         print("acts", acts.shape)
         print("refine acts", acts.reshape(obs1.shape[0],-1,1).shape,  self.qf(obs1).shape)
         print("rews", rews.shape)
         print("done", done.shape)

      # Prediction Q(s)
      acts = acts.reshape(obs1.shape[0],-1,1)
      q = self.qf(obs1).gather(2, acts.long()).squeeze(-1)
      
      # Target for Q regression
      if self.args.algo == 'dqn':      # DQN
         q_target = self.qf_target(obs2)
      elif self.args.algo == 'ddqn':   # Double DQN
         with torch.no_grad():
            argmax = torch.argmax(self.qf(obs2), dim = 2)
            q_target = self.qf_target(obs2)
            q_target = q_target.gather(2, argmax.unsqueeze(2)).squeeze(-1)
            q_target = q_target.mean(1, keepdim = False)
         #q2 = self.qf(obs2)
         #q_target = self.qf_target(obs2)
         #q_target = q_target.gather(1, q2.max(1)[1].unsqueeze(1))
      #q_backup = rews + self.gamma*(1-done)*q_target.max(1)[0]
      q_backup = rews + self.gamma*(1-done)*q_target
      q_backup = torch.stack([q_backup for _ in range(self.act_dim)], dim = 1)
      q_backup.to(self.device)

      if 0: # Check shape of prediction and target
         print("q", q.shape)
         print("q_backup", q_backup.shape)

      # Update perdiction network parameter
      qf_loss = F.mse_loss(q, q_backup.detach())
      self.qf_optimizer.zero_grad()
      qf_loss.backward()
      for param in self.qf.parameters():
        param.grad *= 1/(self.act_dim + 1)
        param.grad.data.clamp_(-1, 1)
      self.qf_optimizer.step()

      # Synchronize target parameters 𝜃‾ as 𝜃 every C steps
      if self.steps % self.target_update_step == 0:
         hard_target_update(self.qf, self.qf_target)
      
      # Save loss
      self.q_losses.append(qf_loss.item())

   def run(self, data):
      step_number = 0
      total_reward = 0.

      obs = np.zeros(7)
      obs[0] = data['encode_fps']
      obs[1] = data['network_fps']
      obs[2] = data['big_temp']/1000
      obs[3] = data['modem_temp']/1000
      obs[4] = data['big_clock2']/10000
      obs[5] = data['res']/100
      obs[6] = data['bitrate']

      if (obs[3] >= 53):
         done = True
         self.end = True
      else:
         done = False
      
      # Keep interacting until agent reaches a terminal state.
      #while not (done or step_number == max_step):

      obs = np.expand_dims(obs, axis = 0)

      if self.eval_mode:
         q_value = self.qf(torch.Tensor(obs).to(self.device))
         q_argmax = torch.argmax(q_value.squeeze(0), dim = 1)
         action = q_argmax.detach().cpu().numpy()    
         #next_obs, reward, done, _ = self.env.step(action)
      else:
         # Collect experience (s, a, r, s') using some policy
         action = self.select_action(torch.Tensor(obs).to(self.device), data)

         # This code is for test (220610)
         '''
         if self.steps % 6 == 0:
            action = np.array([1,1])
         else:
            action = np.array([0,1])
         '''

         #next_obs, reward, done, _ = self.env.step(action)
         reward = self.reward_calculator(data)
         # Add experience to replay buffer
         if self.steps >= 1:
            print('Timestep:', self.steps*2, 'Prev obs: ', self.prev_obs, " Next obs: ", obs, " Action: ", self.prev_action - 1, 
            " Reward: ", reward, " LossQ: ", round(np.mean(self.q_losses), 5))

            self.replay_buffer.add(self.prev_obs, self.prev_action, reward, obs, done)

            ## delayed penalty reward for high temperature
            sample_to_save = {}
            sample_to_save['prev_obs'] = self.prev_obs
            sample_to_save['action'] = self.prev_action
            sample_to_save['reward'] = reward - 3
            sample_to_save['next_obs'] = obs
            sample_to_save['done'] = done
            self.save_current_samples(sample_to_save) 

            if self.prev_obs[0][3] <51 and obs[0][3] >= 51:
               print('Timestep:', self.steps*2, 'Delayed penalty reward for high temperature activated!')
               for  data in self.current_buffer:
                  if data != None:
                     self.replay_buffer.add(data['prev_obs'], data['action'], data['reward'], data['next_obs'], data['done'])
         
         # Start training when the number of experience is greater than batch_size
         if self.steps > self.batch_size + 1:
            self.train_model()

      total_reward += reward
      step_number += 1
      
      self.steps += 1
      self.prev_obs = obs
      self.prev_action = action

      # Save logs
      self.logger['LossQ'] = round(np.mean(self.q_losses), 5)
      #return step_number, total_reward

      if self.steps % 50 == 0 or done == True:
         torch.save(self.qf.state_dict(), "./save_model/model" +self.save_name+".pt")
         print("[Save model]")

      return action, reward

   def reward_calculator(self, data):
      # fps reward
      fps_reward = min(data['network_fps']/self.target_fps,1)
      # resolution reward
      res_reward = min(data['res']/self.max_res, 1)
      # bitrate reward
      br_reward = min(data['bitrate']/70, 1)
      #res_reward = min(data['res'/self.target_bitrate], 1)
      qoe_reward = fps_reward + self.beta1*res_reward + self.beta2*br_reward
      
      ## temperature reward
      temp_reward = 0
      if data['modem_temp']/1000 >= 51:
         temp_reward = -2

      return qoe_reward + temp_reward

   def save_current_samples (self, data):
      self.current_buffer[self.current_buffer_index] = data
      self.current_buffer_index += 1
      self.current_buffer_index %= self.max_current_buffer_size


class agent_runner:
   def __init__(self):
      args = easydict.EasyDict({
         "algo": 'ddqn',
         "load_saved": False,
         "load_path": None,
         "gpu_index": 0
      })
      device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
      #device = torch.device('cpu')
      obs_dim = 7
      act_dim = 3
      act_num = 3

      self.agent = Agent (args, device, obs_dim, act_dim ,act_num)
      print('Start to run RL agent')

   def run_train(self, data):
      if self.agent.end == False:
         action, reward = self.agent.run(data)
         client_action = action - 1
         #print('Obs: ', data, 'Action: ', client_action, 'Reward: ', reward)
         return client_action
      else:
         return np.array([0,0,0])