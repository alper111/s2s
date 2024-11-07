import sys

from agent import Agent

agent = Agent(sys.argv[1])
agent.train_abstraction()
agent.learn_symbols()
