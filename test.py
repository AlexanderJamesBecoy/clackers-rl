import clackers_rl
import gym
import time

env = gym.make('clackers_rl/Single_Pendulum-v0', render_mode="human")
T = 100

a = 0.0
obs, info = env.reset()

for _ in range(T):
    # a = env.action_space.sample()
    # if info["position"] >= 0.0:
    #     a = 2.0
    # else:
    #     a = -2.0
    print(a)
    print(info["position"])
    obs, _, _, _, info = env.step(a)
    time.sleep(1.0)