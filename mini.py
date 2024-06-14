from environment import MNISTSokoban


if __name__ == "__main__":
    # env = MNISTSokoban(map_file="map1.txt", max_crates=2, max_steps=200, render_mode="human")
    env = MNISTSokoban(size=(5, 5), max_crates=3, max_steps=50, render_mode="human",
                       rand_digits=True, rand_agent=True, rand_x=True)
    for _ in range(2000):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, rew, term, trun, info = env.step(action)
            done = term or trun
