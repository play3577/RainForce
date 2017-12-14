# RainForce
Reinforcement Learning Agents in C# (Dynamic Programming, Temporal Difference, Deep Q-Learning, Stochastic/Deterministic Policy Gradients)

**RainForce** is the proper .Net port of [reinforce.js](https://github.com/mbithy/reinforcejs) In particular, the library currently includes:

- **Dynamic Programming** methods ***FULLY PORTED***
- (Tabular) **Temporal Difference Learning** (SARSA/Q-Learning) <<<_in progress_>>>>
- **Deep Q-Learning** for Q-Learning with function approximation with Neural Networks  ***FULLY PORTED***
- **Stochastic/Deterministic Policy Gradients** <<<_I woun't even touch this mess_>>>>

# Code Sketch

A typical usage might look something like:

```C#
var rnd = new Random();
var state = new [] {
 rnd.Next(min, max), rnd.Next(min, max), rnd.Next(min, max), rnd.Next(min, max)
};
var opt = new TrainingOptions {
 Alpha = 0.001,
  Epsilon = 0.5,
  ErrorClamp = 0.002,
  ExperienceAddEvery = 10,
  ExperienceSize = 500,
  ExperienceStart = 0,
  HiddenUnits = 5,
  LearningSteps = 100
};
//we take 4 states 
//we have 2 actions
var agent = new DQNAgent(opt, state.Length, 2);
//get action
var action = agent.Act(state);
//reward result
agent.Learn(1);
```

You can find out more about **Reinforcement Learning Agents** on [karpathy's website](http://cs.stanford.edu/people/karpathy/reinforcejs).

# License

MIT.

# Notes

I'll try where posible to C#-pify this JS lib but the things I have seen this far..... :dizzy_face: :sweat:

