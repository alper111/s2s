import sys
import yaml

from agent import Agent

config = yaml.safe_load(open(sys.argv[1], "r"))
agent = Agent(config)
agent.train_abstraction()
agent.fit_s2s()
