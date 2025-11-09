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
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"
import wandb 
wandb.login()
import torch
import gymnasium as gym
import numpy as np
from utilis.config import ARGConfig
from utilis.default_config import default_config
from model.algo import flowAC
from utilis.Replaybuffer import ReplayMemory
from utilis.video import recorder
import datetime
import itertools
from torch.utils.tensorboard import SummaryWriter
import shutil
from envs.dm_control import DMControlEnv
# from humanoid_bench.env import ROBOTS, TASKS

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True

def evaluation(agent, env, total_numsteps, writer, best_reward, video_path=None):
    avg_reward = 0.
    avg_success = 0.
    if video_path is not None:
        eval_recoder = recorder(video_path)
        eval_recoder.init(f'{total_numsteps}.mp4', enabled=True)
    else:
        eval_recoder = None
    for _  in range(config.eval_times):
        state = env.reset()
        if eval_recoder is not None:
            eval_recoder.record(env.render())
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)

            next_state, reward, done,  info = env.step(action)
            if eval_recoder is not None:
                eval_recoder.record(env.render())
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
        if 'solved' in info.keys():
            avg_success += float(info['solved'])
        elif 'success' in info.keys():
            avg_success += float(info['success'])
    avg_reward /= config.eval_times
    avg_success /= config.eval_times

    if eval_recoder is not None :
        eval_recoder.release('%d_%d.mp4'%(total_numsteps, int(avg_reward)))

    wandb.log(data = {'test/reward': avg_reward,
               'test/avg_success':avg_success}, step=total_numsteps)
    print("----------------------------------------")
    print("Env: {}, Test Episodes: {}, Avg. Reward: {}, Avg. Success: {}".format(config.task, config.eval_times, round(avg_reward, 2), round(avg_success, 2)))
    print("----------------------------------------")
    
    return avg_reward

def train_loop(config, msg = "default"):
    # for humanoid_bench 
    # env = gym.make(config.task)
    # for dm control
    env = DMControlEnv(domain_name=config.domain, task_name=config.task)
    env.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    env.action_space.seed(config.seed)
    np.random.seed(config.seed)
    # set seed
    # Agent
    agent = flowAC(env.observation_space.shape[0], env.action_space, config)

    result_path = './results/{}/{}/{}/{}_{}_{}'.format(config.task, config.algo, msg, 
                                                      datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                                      config.policy, config.seed)

    checkpoint_path = result_path + '/' + 'checkpoint'
    video_path = result_path + '/eval_video'
    log_filename = result_path + '/' + 'training_log.txt'
    # training logs
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    with open(os.path.join(result_path, "config.log"), 'w') as f:
        f.write(str(config))

    writer = SummaryWriter(result_path)
    current_path = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(current_path)
    files_to_save = ['main.py', 'envs', 'model', 'utilis', 'render_results.py']
    ignore_files = [x for x in files if x not in files_to_save]
    # shutil.copytree('.', result_path + '/code', ignore=shutil.ignore_patterns(*ignore_files))

    # memory
    memory = ReplayMemory(config.replay_size, config.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    best_reward = -1e6
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        # state = env.reset()

        while not done:
            if config.start_steps > total_numsteps:
                # action = env.action_space.sample()  # Sample random action
                action = env._action_space.sample()
            else:
                action = agent.select_action(state)  # Sample action from policy

            if config.start_steps <= total_numsteps:
                # Number of updates per step in environment
                for i in range(config.updates_per_step):
                    # Update parameters of all the networks
                    agent.update_parameters(memory, config.batch_size, updates)
                    updates += 1

            next_state, reward, done, info = env.step(action) # Step
            # done = done or truncated
            # next_state, reward, done, info = env.step(action) # Step
            # done = done or truncated
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            if "_max_episode_steps" in dir(env):
                mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            else:
                mask = 1 if episode_steps == 1000 else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory
            state = next_state
            
            # test agent
            if total_numsteps % config.eval_numsteps == 0 and config.eval is True:
                video_path = None
                avg_reward = evaluation(agent, env, total_numsteps, writer, best_reward, video_path)
                if avg_reward >= best_reward and config.save is True:
                    best_reward = avg_reward
                    agent.save_checkpoint(checkpoint_path, 'best')

        if total_numsteps > config.num_steps:
            break

        wandb.log({'train/reward': episode_reward}, step=total_numsteps)
        if i_episode % 10 == 0:

            log_message = "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}\n".format(
                    i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
                )
            with open(log_filename, 'a') as log_file:
                log_file.write(log_message)
    env.close() 
    wandb.finish()



if __name__ == "__main__":
    arg = ARGConfig()
    arg.add_arg("domain","dog","dm domain")
    arg.add_arg("task", "run", "Environment name")
    arg.add_arg("device", 0, "Computing device")
    arg.add_arg("algo", "flowac", "choose algo")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("seed", 0, "experiment seed")
    arg.add_arg("epsilon", 0.0, "random noise for exploration")
    arg.add_arg("lamda", 0.1, "lagrange_multiplier")
    arg.add_arg("solver", "ODE", "select solver between 'ODE' and 'Riemannian'")
    arg.add_arg("path", "CondOT", "select prob path among 'CondOT', 'Affine' and 'Geodesic'")
    arg.add_arg("sch", "CondOT", "select scheduler between 'Linear' and 'CondOT'")
    arg.add_arg("manifold", "Sphere", "select manifold between 'Sphere' and 'Torus'")
    arg.add_arg("method", "midpoint", "select method among torchdiffeq methods")
    arg.parser()
    
    assert arg.solver in ['ODE', 'Riemannian'], f"Invalid solver: {arg.solver}. Must be 'ODE' or 'Riemannian'."
    assert arg.path in ['CondOT', 'Affine', 'Geodesic'], f"Invalid path: {arg.path}. Must be 'CondOT', 'Affine', or 'Geodesic'."
    assert arg.sch in ['Linear', 'CondOT'], f"Invalid sch: {arg.sch}. Must be 'Linear' or 'CondOT'."
    assert arg.manifold in ['Sphere', 'Torus'], f"Invalid manifold: {arg.manifold}. Must be 'Sphere' or 'Torus'."
    assert arg.method in ['dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun', 'euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams']
    config = default_config
    
    config.update(arg)
    wandb.init(
        project = config.algo,
        name = config.domain+ config.task + str(config.seed) + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        config = config
    )
    print(f">>>> Training {config.algo} on {config.task} environment, on {config.device}")
    train_loop(config, msg=config.tag)
