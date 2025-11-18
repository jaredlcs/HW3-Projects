import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DodgerEnv(gym.Env):
    """
    Simple pygame environment for PPO training.

    Agent: rectangle at bottom, moves left/right.
    Obstacle: rectangle falls from top, random x, constant speed.

    Observation: [agent_x_norm, obs_x_norm, obs_y_norm, obs_vy_norm]
    Action space: 0=left, 1=stay, 2=right

    Score: number of frames survived in the current episode.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        width: int = 400,
        height: int = 600,
    ):
        super().__init__()

        assert render_mode in (None, "human", "none")
        self.render_mode = render_mode
        self.width = width
        self.height = height

        # --- RL spaces ---
        # Observation: 4 floats in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        # Actions: left, stay, right
        self.action_space = spaces.Discrete(3)

        # --- Game parameters ---
        self.agent_width = 40
        self.agent_height = 20
        self.agent_speed = 6

        self.obstacle_width = 40
        self.obstacle_height = 20
        self.obstacle_speed = 5

        self.max_steps = 1000

        # Internal state
        self.agent_x = None
        self.agent_y = None
        self.obstacle_x = None
        self.obstacle_y = None
        self.obstacle_vy = None
        self.steps = 0
        self.score = 0  # visible game score (frames survived)

        # Pygame stuff
        self.window = None
        self.clock = None
        self.font = None

        # Colors
        self.BG_COLOR = (30, 30, 30)
        self.AGENT_COLOR = (50, 200, 50)
        self.OBSTACLE_COLOR = (200, 50, 50)
        self.TEXT_COLOR = (255, 255, 255)

    # ------------- Gym API -------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Agent starts centered at bottom
        self.agent_x = self.width // 2
        self.agent_y = self.height - 40

        # Obstacle starts at top with random x
        self.obstacle_x = self.np_random.integers(
            self.obstacle_width // 2,
            self.width - self.obstacle_width // 2,
        )
        self.obstacle_y = 0
        self.obstacle_vy = self.obstacle_speed

        self.steps = 0
        self.score = 0  # reset score at episode start

        obs = self._get_obs()
        info = {"score": self.score}
        return obs, info

    def step(self, action):
        """
        action: int in {0,1,2}
        returns: obs, reward, terminated, truncated, info
        """

        # --- Apply action to agent ---
        if action == 0:  # left
            self.agent_x -= self.agent_speed
        elif action == 2:  # right
            self.agent_x += self.agent_speed
        # action == 1: stay

        # Clamp agent position inside screen
        self.agent_x = np.clip(
            self.agent_x,
            self.agent_width // 2,
            self.width - self.agent_width // 2,
        )

        # --- Move obstacle ---
        self.obstacle_y += self.obstacle_vy

        # If obstacle goes off screen, respawn at top
        if self.obstacle_y > self.height + self.obstacle_height:
            self._respawn_obstacle()

        self.steps += 1
        self.score += 1  # 1 point per frame survived

        # --- Compute reward and done ---
        # Base survival reward
        reward = 1.0

        # Distance-based shaping (helps PPO learn faster)
        distance = abs(self.agent_x - self.obstacle_x)
        max_distance = self.width / 2
        distance_norm = distance / max_distance  # in [0, 1]
        reward += 0.1 * distance_norm  # small bonus for being farther away

        terminated = False
        truncated = False

        # Collision penalty overrides other reward
        if self._check_collision():
            reward = -100.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {"score": self.score}

        # --- Render if needed ---
        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None

    # ------------- Helpers -------------

    def _get_obs(self):
        # Normalize positions / speeds to [0,1]
        agent_x_norm = self.agent_x / self.width
        obs_x_norm = self.obstacle_x / self.width
        obs_y_norm = self.obstacle_y / self.height
        # Cap /normalize speed, assuming <= 20 pixels/frame
        obs_vy_norm = np.clip(self.obstacle_vy / 20.0, 0.0, 1.0)

        obs = np.array(
            [agent_x_norm, obs_x_norm, obs_y_norm, obs_vy_norm],
            dtype=np.float32,
        )
        return obs

    def _check_collision(self):
        # Axis-aligned bounding box collision
        agent_rect = pygame.Rect(
            self.agent_x - self.agent_width // 2,
            self.agent_y - self.agent_height // 2,
            self.agent_width,
            self.agent_height,
        )
        obs_rect = pygame.Rect(
            self.obstacle_x - self.obstacle_width // 2,
            self.obstacle_y - self.obstacle_height // 2,
            self.obstacle_width,
            self.obstacle_height,
        )
        return agent_rect.colliderect(obs_rect)

    def _respawn_obstacle(self):
        self.obstacle_x = self.np_random.integers(
            self.obstacle_width // 2,
            self.width - self.obstacle_width // 2,
        )
        self.obstacle_y = -self.obstacle_height
        self.obstacle_vy = self.obstacle_speed

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("DodgerEnv")
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            # init font once
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 32)

        # Process events so window doesn’t freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.window.fill(self.BG_COLOR)

        # Draw agent
        pygame.draw.rect(
            self.window,
            self.AGENT_COLOR,
            pygame.Rect(
                self.agent_x - self.agent_width // 2,
                self.agent_y - self.agent_height // 2,
                self.agent_width,
                self.agent_height,
            ),
        )

        # Draw obstacle
        pygame.draw.rect(
            self.window,
            self.OBSTACLE_COLOR,
            pygame.Rect(
                self.obstacle_x - self.obstacle_width // 2,
                self.obstacle_y - self.obstacle_height // 2,
                self.obstacle_width,
                self.obstacle_height,
            ),
        )

        # Draw score
        if self.font is not None:
            score_surf = self.font.render(f"Score: {self.score}", True, self.TEXT_COLOR)
            self.window.blit(score_surf, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


if __name__ == "__main__":
    env = DodgerEnv(render_mode="human")
    obs, info = env.reset()

    done = False
    truncated = False

    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            print("Episode finished. Score:", info.get("score", None))
            obs, info = env.reset()
