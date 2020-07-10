# Dexterous Gym - Challenging Extensions to OpenAI's Gym Hand Manipulation Environments
This repository contains a number of challenging environments based on OpenAI's Gym hand manipulation tasks, all of which are challenging for current RL and trajectory optimisation techniques. The GIFs of example solutions for some of the environments come from a soon-to-be published custom trajectory optimisation technique I have developed.

## Installation
Requirements: [Mujoco-py](https://github.com/openai/mujoco-py) and [Gym](https://github.com/openai/gym)
Once these are installed, run `pip install dexterous-gym`.
Alternatively, clone the repository and run `pip install -e .` from within the main directory.

## Environments
### Standard RL environments
<img src="dexterous_gym/examples/penspin.gif" align="right" width="250" border="1"/>
We include one "standard" RL environment (i.e. an environment which is not conditioned on a goal), PenSpin.

\
<b>Variants</b>: "PenSpin-v0". 

This is a simple modification of the OpenAI gym HandPen environment where all we do is change the observation space to remove any notion of a goal and then define a custom reward function. This reward function encourages the pen to be spun whilst remaining horizontal.

### Goal-based environments
A  majority of the environments are goal-based, and have a similar API to the openAI Gym manipulation environments (observations are dictionaries with "observation", "achieved_goal", "desired_goal"). In the case of the two object environments the "achieved_goal" and "desired_goal" entries are also dictionaries with entries "object_1" and "object_2". All environments come with a standard setting (where the reward is defined in terms of the positional and rotational distance between the achieved goal(s) and the desired goal(s) as well as a sparse setting, where the reward is -1.0 until the goal(s) are achieved (where the reward is 0.0).

#### HandOver Environments
<img src="dexterous_gym/examples/egghandover.gif" align="right" width="250" border="1"/>
<b>Variants</b>: "EggHandOver-v0", "BlockHandOver-v0", "PenHandOver-v0", "EggHandOverSparse-v0", "BlockHandOverSparse-v0", "PenHandOverSparse-v0" <br/><br/>
These environments involve two fixed shadow hands. The hand which starts with the object must find a way to hand it over to the second hand which can then move it to the desired goal.  

#### HandCatchUnderarm Environments
These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained regions. The aims of these tasks is to throw the object from one hand to the other, which can then move it to the desired position/rotation. The "Hard" versions just mean that there is a greater distance between the two hands, such that the objects must be thrown further.
