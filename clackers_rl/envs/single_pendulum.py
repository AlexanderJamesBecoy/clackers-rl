import gym
from gym import spaces
import pygame
import numpy as np

_N_STATES = 2   # angular position, angular velocity.
_N_ACTIONS = 2  # clockwise and counter-clockwise torques.
_N_STEPS = 51   # Number of discretization of each state.

_STATE_POS = 0  # Index for state angular position.
_STATE_POS_MIN = -np.pi # Lower bound of the angular position.
_STATE_POS_MAX = np.pi  # Upper bound of the angular position.
_STATE_VEL = 1  # Index for state angular velocity.
_STATE_VEL_MIN = -np.pi * 5.0   # Lower bound of the angular velocity.
_STATE_VEL_MAX = np.pi * 5.0    # Upper bound of the angular velocity.

_STATES_MIN = np.array([_STATE_POS_MIN, _STATE_VEL_MIN])
_STATES_MAX = np.array([_STATE_POS_MAX, _STATE_VEL_MAX])

_TORQUE_MIN = -np.pi * 5.0  # Lower bound of the external torque.
_TORQUE_MAX = np.pi * 5.0   # Upper bound of the external torque.

class SinglePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self, render_mode=None, size=5):
        self.size = size # The size of the square grid
        self.window_size = 512 # The size of the PyGame window

        # Rendering variables.
        self._pivot_pos = (self.window_size / 2, self.window_size / 3)
        self._mass_pos = (self.window_size / 2, 2 * self.window_size / 3)
        self._rod_length = self.window_size / 3

        # Observations are the states of the single rigid pendulum.
        # First state is the angular position, the second is the angular velocity.
        self.observation_space = spaces.Box(_STATES_MIN, _STATES_MAX, dtype=np.float64)

        # We have 2 actions, corresponding to "clockwise" and "counter-clockwise" torque in Nm.
        self.action_space = spaces.Box(_TORQUE_MIN, _TORQUE_MAX, dtype=np.float64)

        # Mechanical properties of the single rigid pendulum.
        self._m = 1     # kg, mass of the ball. Assume link is massless.
        self._l = 1     # m, length of the link.
        self._g = 9.81  # m/s, gravitational acceleration.
        self._dt = 0.1  # s, instantaneous change in time.

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.winow` will be a reference
        to the winow that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._states

    def _get_info(self):
        return {"position": self._states[_STATE_POS], "velocity": self._states[_STATE_VEL]}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize the continuous states to random legal values.
        self._states = np.zeros((_N_STATES,))
        self._states[_STATE_POS] = np.random.uniform(_STATE_POS_MIN, _STATE_POS_MAX)
        # self._states[_STATE_VEL] = np.random.uniform(_STATE_VEL_MIN, _STATE_VEL_MAX)

        # Initialize the continuous torque equal to zero.
        self._torque = 0.0

        observation = self._get_obs()
        info = self._get_info()

        # Obtain the initial 2D Cartesian coordinates of the mass.
        self._mass_pos = self._mass_coordinates(self._states[_STATE_POS])

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        terminated = False
        reward = 1 if terminated else 0 # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        # Apply inverse dynamics to obtain the angular acceleration w.r.t. external torque input.
        acc = self._inverse_dynamic(action)

        # Perform forward dynamics to obtain the angular position and angular velocity.
        vel = self._states[_STATE_VEL] + acc * self._dt
        pos = self._states[_STATE_POS] + vel * self._dt
        pos = self._normalize_position(pos)

        self._states = np.array([pos, vel])
        self._mass_pos = self._mass_coordinates(pos)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw the pivot of the pendulum as a black dot.
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            self._pivot_pos,
            10,
        )

        # Draw the rod of the pendulum as a black line.
        pygame.draw.line(
            canvas,
            (0, 0, 0),
            self._pivot_pos,
            self._mass_pos,
            5,
        )

        # Draw the massive bob of the pendulum as a gray circle.
        pygame.draw.circle(
            canvas,
            (155, 155, 155),
            self._mass_pos,
            20,
        )
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:   # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _inverse_dynamic(self, torque_ext):
        """
        Calculate the acceleration of the single rigid pendulum.
        TODO: Add friction component.
        """
        pos = self._states[_STATE_POS]

        torque = self._torque + torque_ext
        g_conv = (self._g * np.cos(pos)) / self._l

        acc = g_conv + torque / (self._m * self._l**2)

        return acc
    
    def _discretize_states(self):
        """
        Discretize the states such that the Q-table and calculation is tractable.
        """
        pass

    def _mass_coordinates(self, theta):
        """
        Calculate the absolute 2D Cartesian coordinates of the massive bob using
        the angular position theta.
        """
        x = self._rod_length * np.cos(theta) + self._pivot_pos[0]
        y = self._rod_length * np.sin(theta) + self._pivot_pos[1]
        
        return (x, y)
    
    def _normalize_position(self, theta):
        """
        Normalize the angular position into an angle between -pi and pi.
        TODO: Fix this because without normalization -> theta goes (+-) infinity
        """
        theta = np.mod(theta, 2 * np.pi)
        return theta