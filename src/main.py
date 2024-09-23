import yaml

from agent import Agent

config = yaml.safe_load(open("config.yaml", "r"))
agent = Agent(config)
agent.train_abstraction()
agent.fit_s2s()
