from gym.envs.registration import register

register(
    id='HopperFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Hopper-v2'
    )
)

register(
    id='SwimmerMod-v1',
    entry_point='envs.mujocoFH:SwimmerEnv'
)

register(
    id='SwimmerFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='SwimmerMod-v1'
    )
)

register(
    id='HopperInverse-v0',
    entry_point='envs.mujocoFH:HopperInverse',
    kwargs=dict(
        env_name='Hopper-v2'
    )
)

register(
    id='Walker2dFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Walker2d-v2'
    )
)

register(
    id='HalfCheetahFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='HalfCheetah-v2'
    )
)

register(
    id='AntFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Ant-v2'
    )
)

register(
    id='HumanoidFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Humanoid-v2'
    )
)

