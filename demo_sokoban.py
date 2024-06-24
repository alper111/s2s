from environments.sokoban import MNISTSokoban


if __name__ == "__main__":
    # env = MNISTSokoban(map_file="map1.txt", max_crates=2, max_steps=200, render_mode="human")
    all_random = False
    env = MNISTSokoban(size=(5, 5), max_crates=2, max_steps=50, object_centric=False,
                       render_mode="human", rand_digits=all_random, rand_agent=all_random, rand_x=all_random)
    for _ in range(2000):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, rew, term, trun, info = env.step(action)
            done = term or trun
