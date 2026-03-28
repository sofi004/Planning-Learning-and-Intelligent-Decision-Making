import random  # random module for noise and stochastic events

# --- Game Constants ---
GRAVITY = 0.15  # downward acceleration applied each tick when not thrusting
THRUST = 0.35  # upward impulse applied when the agent thrusts
CANVAS_HEIGHT = 500  # vertical canvas size in pixels
BAR_SIZE = 80  # height of the green bar in pixels
FISH_SIZE = 20  # fish visual size (unused in logic but available)
WIN_THRESHOLD = 5.0  # seconds inside required to win (accumulated)
LOSE_THRESHOLD = -5.0  # seconds outside required to lose (accumulated)


class FishType:
    """Container for fish movement parameters used to vary difficulty."""

    def __init__(self, name, min_y, max_y, noise_par, jump_prob, jump_size):
        self.name = name  # human-readable fish name
        self.min_y = min_y  # min allowed vertical position for the fish
        self.max_y = max_y  # max allowed vertical position for the fish
        self.noise_par = noise_par  # std dev used for velocity noise
        self.jump_prob = jump_prob  # probability per tick of a random jump
        self.jump_size = jump_size  # magnitude of jump impulse


FISH_TYPES = [
    FishType("Carp", 100, 400, 0.5, 0.001, 20),
    FishType("Trout", 50, 450, 1.5, 0.01, 40),
    FishType("Salmon", 50, 450, 2.0, 0.02, 50),
    FishType("Catfish", 300, 480, 1.0, 0.005, 30),
    FishType("Pike", 50, 450, 2.5, 0.03, 60),
    FishType("Sturgeon", 200, 400, 1.2, 0.002, 25),
    FishType("Tuna", 50, 450, 3.5, 0.04, 70),
    FishType("Shark", 50, 450, 2.0, 0.05, 40),
    FishType("Legend", 20, 480, 4.0, 0.08, 90),
    FishType("The Glitch", 0, 500, 6.0, 0.15, 120),
]  # predefined fish types with different dynamics


class FishingGameLogic:
    """Core game physics and simple reward logic used by agents and runners."""

    def __init__(self, fish_name=None):
        self.reset_game(fish_name)  # initialize/reset game state

    def reset_game(self, fish_name=None):
        # Pick a specific fish if requested, otherwise a random one
        if fish_name:
            fish = next((f for f in FISH_TYPES if f.name == fish_name), None)
            if fish:
                self.current_fish = fish
            else:
                self.current_fish = random.choice(FISH_TYPES)
        else:
            self.current_fish = random.choice(FISH_TYPES)

        # Bar initial state
        self.bar_y = 400.0  # bar vertical position
        self.bar_vel = 0.0  # bar vertical velocity

        # Fish initial state in middle of its allowed range
        self.fish_y = (self.current_fish.min_y + self.current_fish.max_y) / 2
        self.fish_vel = 0.0  # fish vertical velocity

        # Catch timer represents cumulative 'inside' vs 'outside' time
        self.catch_timer = 0.0
        self.game_running = True  # flag that the game is active

    def step_physics(self, action):
        """Advance one tick: apply action, update physics, return (state, reward, done)."""
        if not self.game_running:
            return self.get_state(), 0, True  # if game over, return terminal

        # 1. Update bar (player-controlled) physics
        if action:
            self.bar_vel -= THRUST  # thrust reduces bar_y (move up)
        else:
            self.bar_vel += GRAVITY  # gravity pulls bar down

        self.bar_y += self.bar_vel  # integrate velocity into position

        # 2. Update fish with noisy velocity dynamics
        move_noise = random.gauss(0, self.current_fish.noise_par)  # gaussian noise
        self.fish_vel += move_noise  # addnoise to velocity

        # Random jump impulse occasionally
        if random.random() < self.current_fish.jump_prob:
            jump_impulse = self.current_fish.jump_size * random.choice([-1, 1])
            self.fish_vel += jump_impulse

        # Damping for smooth movement
        self.fish_vel *= 0.85

        # Update fish position from velocity
        self.fish_y += self.fish_vel

        # Clamp fish within its allowed min/max
        self.fish_y = max(self.current_fish.min_y, min(self.fish_y, self.current_fish.max_y))

        # Bounce if exactly at boundaries (simple reflection)
        if self.fish_y == self.current_fish.min_y or self.fish_y == self.current_fish.max_y:
            self.fish_vel *= -0.5

        # 3. Clamp bar within canvas
        self.bar_y = max(0, min(self.bar_y, CANVAS_HEIGHT - BAR_SIZE))
        if self.bar_y == 0 or self.bar_y == CANVAS_HEIGHT - BAR_SIZE:
            self.bar_vel = 0  # stop velocity at hard bounds

        # 4. Compute reward: inside the bar -> negative cost, outside -> positive cost
        inside = self.bar_y <= self.fish_y <= self.bar_y + BAR_SIZE

        step_cost = -1.0 if inside else 1.0  # cost formulation (negative when succeeding)

        # Update catch timer (small timestep increment ~ 16ms per frame)
        if inside:
            self.catch_timer += 0.016
        else:
            self.catch_timer -= 0.016

        done = False

        # Win/lose when catch_timer passes thresholds
        if self.catch_timer >= WIN_THRESHOLD or self.catch_timer <= LOSE_THRESHOLD:
            done = True
            self.game_running = False

        return self.get_state(), step_cost, done

    def get_state(self):
        """Return minimal state dict used by agents: fish_y, bar_y, bar_vel."""
        return {"fish_y": self.fish_y, "bar_y": self.bar_y, "bar_vel": self.bar_vel}

    def get_fish_name(self):
        """Return the name of the current fish type."""
        return self.current_fish.name
