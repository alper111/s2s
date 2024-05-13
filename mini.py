from environment import MNISTSokoban

env = MNISTSokoban(size=(7, 7), max_crates=2, max_steps=200, render_mode="human")
for _ in range(2000):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, rew, term, trun, info = env.step(action)
        done = term or trun

