# Reinforcement Learning

**Reinforcement Learning (RL)** is a powerful branch of Machine Learning designed to solve problems where decisions are made sequentially. In RL, machines learn from their actions, using the data observed up to time *t* to decide which action to take at time *t + 1*. This approach is also used in Artificial Intelligence to train agents for complex tasks, such as walking or playing games. The goal is to maximize cumulative rewards: desired outcomes provide a reward, while undesired outcomes incur a penalty. Machines learn through a process of trial and error to optimize their actions over time.

By the end of this section, you will understand how to implement and apply Reinforcement Learning algorithms to decision-making problems, providing insights into how intelligent agents can learn and improve their performance through experience.

## Overview of Reinforcement Learning

In this part of the course, we will cover the implementation and interpretation of the following RL models:

- **Upper Confidence Bound (UCB)**: A model that balances exploration and exploitation by selecting actions based on upper confidence estimates of potential rewards.
- **Thompson Sampling**: A probabilistic algorithm that chooses actions by sampling from a posterior distribution, allowing for efficient exploration in uncertain environments.

## Key Concepts

Reinforcement Learning is based on the concept of an agent interacting with an environment to maximize long-term rewards:
- **Reward**: A scalar feedback signal given to the agent after each action, indicating how good or bad the action was.
- **Exploration vs Exploitation**: The trade-off between exploring new actions to discover their potential rewards and exploiting known actions that yield the best rewards.
- **Policy**: The strategy used by the agent to decide which action to take at each time step.

## Use Cases
- **Robotics**: Train machines to perform complex tasks, like walking or object manipulation, by learning optimal control policies.
- **Game AI**: Develop agents that can learn to play games such as chess or Go by optimizing their strategies through trial and error.
- **Autonomous Systems**: Enable self-driving cars or drones to navigate environments by making intelligent decisions in real-time.

A great complementary resource for this chapter is the following book: [AI Crash Course - by Hadelin de Ponteves](https://www.amazon.com/Crash-Course-hands-introduction-reinforcement/dp/1838645357/ref=sr_1_1?crid=235YAFPX03J0Z&dchild=1&keywords=ai+crash+course&qid=1594476675&sprefix=ai+cr%2Caps%2C213&sr=8-1).