# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
import os
import sys 
import cv2
import gym
import time
import torch 
import random
import numpy as np
from collections import deque
from datetime import datetime
from replay_buffer import ReplayBuffer
from agent import TQC
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from helper import FrameStack, mkdir, make_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
import logging

def set_egl_device(device):
    assert "EGL_VISIBLE_DEVICES" not in os.environ, "Do not manually set EGL_VISIBLE_DEVICES"
    try:
        egl_id = get_egl_device_id(cuda_id)
    except Exception:
        logging.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to VREnv README"
        )
   
        egl_id = 0
    os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    # logging.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")

def get_egl_device_id(cuda_id):
    """
    >>> i = get_egl_device_id(0)
    >>> isinstance(i, int)
    True
    """
    assert isinstance(cuda_id, int), "cuda_id has to be integer"
    dir_path = Path(__file__).absolute().parents[2] / "egl_check"
    dir_path = "/home/leiningc"
    if not os.path.isfile(dir_path / "EGL_options.o"):
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("Building EGL_options.o")
            subprocess.call(["bash", "build.sh"], cwd=dir_path)
        else:
            # In case EGL_options.o has to be built and multiprocessing is used, give rank 0 process time to build
            time.sleep(5)
    result = subprocess.run(["./EGL_options.o"], capture_output=True, cwd=dir_path)
    n = int(result.stderr.decode("utf-8").split(" of ")[1].split(".")[0])
    for egl_id in range(n):
        my_env = os.environ.copy()
        my_env["EGL_VISIBLE_DEVICE"] = str(egl_id)
        result = subprocess.run(["./EGL_options.o"], capture_output=True, cwd=dir_path, env=my_env)
        match = re.search(r"CUDA_DEVICE=[0-9]+", result.stdout.decode("utf-8"))
        if match:
            current_cuda_id = int(match[0].split("=")[1])
            if cuda_id == current_cuda_id:
                return egl_id
    raise EglDeviceNotFoundError


def evaluate_policy(policy, writer, total_timesteps, config, env, episode=5):
    """    
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """

    path = mkdir("","eval/" + str(total_timesteps) + "/")
    avg_reward = 0.
    seeds = [x for x in range(episode)]
    goal= 0
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        env.seed(s)
        obs = env.reset()
        done = False
        step = 0
        while True:
            step += 1
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            #cv2.imshow("wi", cv2.resize(obs[:,:,::-1], (300,300)))
            # frame = cv2.imwrite("{}/wi{}.png".format(path, step), np.array(obs))
            if done:
                avg_reward += reward 
                if step < 150:
                    goal +=1
                step = 0
                break
            #cv2.waitKey(10)
            avg_reward += reward 

    avg_reward /= len(seeds)
    writer.add_scalar('Evaluation reward', avg_reward, total_timesteps)
    writer.add_scalar('Evaluation goals', goal, total_timesteps)
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: {}  goal reached {} of  {} ".format(avg_reward, goal, episode))
    print ("---------------------------------------")
    return avg_reward


def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')


def time_format(sec):
    """
    
    Args:
        param1():

    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)


def train_agent(config):
    """
    Args:
    """
    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    file_name = str(config["locexp"]) + "/pytorch_models/{}".format(str(config["env_name"]))    
    pathname = dt_string 
    tensorboard_name = str(config["locexp"]) + '/runs/' + pathname
    print(tensorboard_name)
    writer = SummaryWriter(tensorboard_name)
    size = config["size"]
    #env= gym.make(config["env_name"], renderer='egl')
    # Create the vectorized environment
    cuda = int(os.environ['CUDA_VISIBLE_DEVICES'])
    print("device ", cuda)
    state_dim = 200
    print("State dim, " , state_dim)
    action_dim = 5 
    print("action_dim ", action_dim)
    max_action = 1
    config["target_entropy"] =-np.prod(action_dim)
    obs_shape = (config["history_length"], size, size)
    action_shape = (action_dim,)
    print("obs", obs_shape)
    print("act", action_shape)
    policy = TQC(state_dim, action_dim, max_action, config)    
    replay_buffer = ReplayBuffer(obs_shape, action_shape, int(config["buffer_size"]), config["image_pad"], config["device"])
    if config["continue_training"]:
        print("continue_training")
        replay_buffer.load_memory(config["memory_path"])
        print("load at point replay buffer {}".format(replay_buffer.idx))
        policy.load(config["model_path"])
    set_egl_device(cuda)
    eval_env= gym.make(config["env_name"], renderer='egl')
    eval_env = FrameStack(eval_env, config)
    env = SubprocVecEnv([make_env(config, i) for i in range(int(config['parallel_envs']))])
    obs = env.reset()
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = np.array([1,1,1,1,1,1,1,1])
    t0 = time.time()
    scores_window = deque(maxlen=100) 
    episode_reward = []
    evaluations = []
    tb_update_counter = 0
    evaluations.append(evaluate_policy(policy, writer, total_timesteps, config, eval_env))
    save_model = file_name + '-{}reward_{:.2f}'.format(episode_num, 0) 
    # policy.save(save_model)
    done_counter = deque(maxlen=100)
    text_file = os.path.join(config["res_path"] , str(dt_string))
    time_to_save = False
    already_saved = False
    time_save_buffer = 86000
    if int(config["continue_training"]):
        replay_buffer.load_memory(config["memory_path"])
        print("load at point replay buffer {}".format(replay_buffer.idx))
        policy.load(config["model_path"])
        total_timesteps = config["timestep"]
        skip = config["timestep"]
        episode_num = config["episode_num"]

    while total_timesteps < config["max_timesteps"]:
        tb_update_counter += 1
        # If the episode is done
        if done.any():
            #print(episode_reward)
            episode_reward = np.mean(np.sum(np.array(episode_reward))) / config["parallel_envs"]
            # print(episode_reward)
            episode_reward = np.mean(np.array(episode_reward))
            scores_window.append(episode_reward)
            average_mean = np.mean(scores_window)
            if tb_update_counter > config["tensorboard_freq"]:
                print("Write tensorboard")
                tb_update_counter = 0
                writer.add_scalar('Reward', episode_reward, total_timesteps)
                writer.add_scalar('Reward mean ', average_mean, total_timesteps)
                writer.flush()
            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                if episode_timesteps < 150:
                    done_counter.append(1)
                else:
                    done_counter.append(0)
                goals = sum(done_counter)
                text = "Total Timesteps: {} Episode Num: {} ".format(total_timesteps, episode_num) 
                text += "Episode steps {} ".format(episode_timesteps)
                text += "Goal last 100 ep : {} ".format(goals)
                text += "Reward: {:.2f}  Average Re: {:.2f} Time: {}".format(episode_reward, np.mean(scores_window), time_format(time.time()-t0))
                text += "Replay_buffer_size : {} ".format(replay_buffer.idx)
                writer.add_scalar('Goal_freq', goals, total_timesteps)
                time_passed = int(time.time()- t0)
                if time_passed >= 85800 and not already_saved:
                    print("time to save ")
                    time_to_save = True
                    already_saved = True
                print("time passed ", time_passed)
                print(text)
                print(text_file)
                text_file = config["locexp"] + "/results"
                write_into_file(text_file, text)
            # We evaluate the episode and we save the policy

            # time_to_save = True
            if episode_num % config["eval_freq"] == 0:
                timesteps_since_eval %= config["eval_freq"]
                evaluations.append(evaluate_policy(policy, writer, total_timesteps, config, eval_env))
                save_model = file_name + '-{}reward_{:.2f}'.format(episode_num, evaluations[-1]) 
                policy.save(save_model)
                memory_path = os.path.join(config["locexp"], "save_memory-{}".format(episode_num))
                replay_buffer.save_memory(memory_path)
            # When the training step is done, we reset the state of the environment
            obs = env.reset()
            episode_num += 1

            # Set the Done to False
            done = False
            # Set rewards and episode timesteps to zero
            episode_reward = []
            episode_timesteps = 0
        # Before 10000 timesteps, we play random actions
        if total_timesteps < config["start_timesteps"]:
            action = []
            for i in range(config["parallel_envs"]):
                action.append(env.action_space.sample())
            action = np.array(action)
        else: # After 10000 timesteps, we switch to the model
            action = policy.select_action_batch(obs)
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        #print("action shape ", action.shape)
        new_obs, reward, done, _ = env.step(action)
        # print(reward)
        #frame = cv2.imshow("wi", np.array(new_obs))
        #cv2.waitKey(10)
        
        # We check if the episode is done
        #done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        # We increase the total reward
        episode_reward.append(reward)
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        #print("s ", obs.shape)
        #print(done)
        replay_buffer.add_batch(obs, action, reward, new_obs, done, done)
        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        # print("t", total_timesteps)
        if total_timesteps > config["start_timesteps"]:
            for i in range(config["repeat_update"]):
                policy.train(replay_buffer, writer, 1)
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
