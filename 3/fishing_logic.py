import random

# --- Game Constants ---
GRAVITY = 0.15
THRUST = 0.35
CANVAS_HEIGHT = 500
BAR_SIZE = 80
FISH_SIZE = 20
WIN_THRESHOLD = 5.0  # Seconds inside to win
LOSE_THRESHOLD = -5.0  # Seconds outside to lose


class FishType:
    def __init__(self, name, min_y, max_y, noise_par, jump_prob, jump_size):
        self.name = name
        self.min_y = min_y
        self.max_y = max_y
        self.noise_par = noise_par  # Std dev for random walk velocity/pos
        self.jump_prob = jump_prob  # Probability per frame
        self.jump_size = jump_size


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
]


class FishingGameLogic:
    def __init__(self, fish_name=None):
        self.reset_game(fish_name)

    def reset_game(self, fish_name=None):
        # Pick a random fish or specific one
        if fish_name:
            # Find the fish by name
            fish = next((f for f in FISH_TYPES if f.name == fish_name), None)
            if fish:
                self.current_fish = fish
            else:
                self.current_fish = random.choice(FISH_TYPES)
        else:
            self.current_fish = random.choice(FISH_TYPES)

        # State Variables
        self.bar_y = 400.0
        self.bar_vel = 0.0

        # Fish starts in middle of its range
        self.fish_y = (self.current_fish.min_y + self.current_fish.max_y) / 2
        self.fish_vel = 0.0  # Used for smooth movement with noise

        self.catch_timer = 0.0  # Sentient "Tug of War"
        self.game_running = True

    def step_physics(self, action):
        """
        Advances the game by one tick based on action.
        Returns: (state, reward, done)
        Reward: +1 if inside, -1 if outside
        """
        if not self.game_running:
            return self.get_state(), 0, True

        # 1. Update Bar Physics
        if action:
            self.bar_vel -= THRUST
        else:
            self.bar_vel += GRAVITY

        self.bar_y += self.bar_vel

        # 2. Update Fish (Complex Movement with velocity)
        # Add random noise to velocity instead of position
        move_noise = random.gauss(0, self.current_fish.noise_par)
        self.fish_vel += move_noise

        # Jump mechanic - apply as velocity impulse for smooth animation
        if random.random() < self.current_fish.jump_prob:
            jump_impulse = self.current_fish.jump_size * random.choice([-1, 1])
            self.fish_vel += jump_impulse

        # Apply velocity damping for natural deceleration
        self.fish_vel *= 0.85

        # Update position from velocity
        self.fish_y += self.fish_vel

        # Bounds for fish
        self.fish_y = max(
            self.current_fish.min_y, min(self.fish_y, self.current_fish.max_y)
        )

        # Bounce off boundaries
        if (
            self.fish_y == self.current_fish.min_y
            or self.fish_y == self.current_fish.max_y
        ):
            self.fish_vel *= -0.5

        # 3. Bar Boundaries
        self.bar_y = max(0, min(self.bar_y, CANVAS_HEIGHT - BAR_SIZE))
        if self.bar_y == 0 or self.bar_y == CANVAS_HEIGHT - BAR_SIZE:
            self.bar_vel = 0

        # 4. Check status & Reward
        inside = self.bar_y <= self.fish_y <= self.bar_y + BAR_SIZE

        step_cost = -1.0 if inside else 1.0

        if inside:
            self.catch_timer += 0.016
        else:
            self.catch_timer -= 0.016

        done = False

        if self.catch_timer >= WIN_THRESHOLD or self.catch_timer <= LOSE_THRESHOLD:
            done = True
            self.game_running = False

        return self.get_state(), step_cost, done

    def get_state(self):
        return {"fish_y": self.fish_y, "bar_y": self.bar_y, "bar_vel": self.bar_vel}

    def get_fish_name(self):
        return self.current_fish.name
