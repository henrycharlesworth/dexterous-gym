from gym.envs.registration import register

"""
HandOver environments
"""
register(
    id='EggHandOver-v0',
    entry_point='dexterous_gym.envs.egg_hand_over:EggHandOver',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='BlockHandOver-v0',
    entry_point='dexterous_gym.envs.block_hand_over:BlockHandOver',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='PenHandOver-v0',
    entry_point='dexterous_gym.envs.pen_hand_over:PenHandOver',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='EggHandOverSparse-v0',
    entry_point='dexterous_gym.envs.egg_hand_over:EggHandOver',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='BlockHandOverSparse-v0',
    entry_point='dexterous_gym.envs.block_hand_over:BlockHandOver',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='PenHandOverSparse-v0',
    entry_point='dexterous_gym.envs.pen_hand_over:PenHandOver',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)


"""
Underarm HandCatch envs
"""

register(
    id='EggCatchUnderarm-v0',
    entry_point='dexterous_gym.envs.egg_catch_underarm:EggCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='BlockCatchUnderarm-v0',
    entry_point='dexterous_gym.envs.block_catch_underarm:BlockCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='PenCatchUnderarm-v0',
    entry_point='dexterous_gym.envs.pen_catch_underarm:PenCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='EggCatchUnderarmHard-v0',
    entry_point='dexterous_gym.envs.egg_catch_underarm_hard:EggCatchUnderarmHard',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='BlockCatchUnderarmHard-v0',
    entry_point='dexterous_gym.envs.block_catch_underarm_hard:BlockCatchUnderarmHard',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='PenCatchUnderarmHard-v0',
    entry_point='dexterous_gym.envs.pen_catch_underarm_hard:PenCatchUnderarmHard',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='EggCatchUnderarmSparse-v0',
    entry_point='dexterous_gym.envs.egg_catch_underarm:EggCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='BlockCatchUnderarmSparse-v0',
    entry_point='dexterous_gym.envs.block_catch_underarm:BlockCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='PenCatchUnderarmSparse-v0',
    entry_point='dexterous_gym.envs.pen_catch_underarm:PenCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='EggCatchUnderarmHardSparse-v0',
    entry_point='dexterous_gym.envs.egg_catch_underarm_hard:EggCatchUnderarmHard',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='BlockCatchUnderarmHardSparse-v0',
    entry_point='dexterous_gym.envs.block_catch_underarm_hard:BlockCatchUnderarmHard',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='PenCatchUnderarmHardSparse-v0',
    entry_point='dexterous_gym.envs.pen_catch_underarm_hard:PenCatchUnderarmHard',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='TwoEggCatchUnderArm-v0',
    entry_point='dexterous_gym.envs.two_egg_catch_underarm:TwoEggCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='TwoBlockCatchUnderArm-v0',
    entry_point='dexterous_gym.envs.two_block_catch_underarm:TwoBlockCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='TwoPenCatchUnderArm-v0',
    entry_point='dexterous_gym.envs.two_pen_catch_underarm:TwoPenCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='TwoEggCatchUnderArmSparse-v0',
    entry_point='dexterous_gym.envs.two_egg_catch_underarm:TwoEggCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='TwoBlockCatchUnderArmSparse-v0',
    entry_point='dexterous_gym.envs.two_block_catch_underarm:TwoBlockCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='TwoPenCatchUnderArmSparse-v0',
    entry_point='dexterous_gym.envs.two_pen_catch_underarm:TwoPenCatchUnderarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)


"""
overarm HandCatch envs
"""

register(
    id='EggCatchOverarm-v0',
    entry_point='dexterous_gym.envs.egg_catch_overarm:EggCatchOverarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='BlockCatchOverarm-v0',
    entry_point='dexterous_gym.envs.block_catch_overarm:BlockCatchOverarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='PenCatchOverarm-v0',
    entry_point='dexterous_gym.envs.pen_catch_overarm:PenCatchOverarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'dense'}
)

register(
    id='EggCatchOverarmSparse-v0',
    entry_point='dexterous_gym.envs.egg_catch_overarm:EggCatchOverarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='BlockCatchOverarmSparse-v0',
    entry_point='dexterous_gym.envs.block_catch_overarm:BlockCatchOverarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)

register(
    id='PenCatchOverarmSparse-v0',
    entry_point='dexterous_gym.envs.pen_catch_overarm:PenCatchOverarm',
    max_episode_steps=75,
    kwargs={'reward_type': 'sparse'}
)


"""
PenSpin
"""
register(
    id='PenSpin-v0',
    entry_point='dexterous_gym.envs.pen_spin:PenSpin',
    max_episode_steps=250,
)
