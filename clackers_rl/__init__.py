from gym.envs.registration import register

register(
    id="clackers_rl/Single_Pendulum-v0",
    entry_point='clackers_rl.envs:SinglePendulumEnv',
    max_episode_steps=300,
)