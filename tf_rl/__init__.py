import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------
#jackal envs
# register(
#     id='GazeboJackalLidar-v0',
#     entry_point='tf_rl.envs:GazeboJackalLidarEnv',
#     # More arguments here
# )
# register(
#     id='GazeboJackalLidar-v1',
#     entry_point='tf_rl.envs:GazeboJackalLidarEnvADDPG',
#     # More arguments here
# )
# register(
#     id='GazeboMultiJackalLidar-v1',
#     entry_point='tf_rl.envs:GazeboMultiJackalLidarEnv',
#     # More arguments here
# )
# register(
#     id='GazeboSample-v1',
#     entry_point='tf_rl.envs:GazeboSampleEnv',
#     # More arguments here
# )
# register(
#     id='GazeboMultiJackalLidar-v2',
#     entry_point='tf_rl.envs:GazeboMultiJackalLidarEnvADDPG',
#     # More arguments here
# )
#
# register(
#     id='GazeboMultiJackalLidar-v3',
#     entry_point='tf_rl.envs:GazeboMultiJackalLidarEnv3',
#     # More arguments here
# )
#
# register(
#     id='GazeboMultiJackalLidar-v4',
#     entry_point='tf_rl.envs:GazeboMultiJackalLidarEnv4',
#     # More arguments here
# )
# register(
#     id='GazeboMultiJackalLidarTest-v4',
#     entry_point='tf_rl.envs:GazeboMultiJackalLidarTestEnv4',
#     # More arguments here
# )
#
#
#
#
# register(
#     id='GazeboSinglePPo-v0',
#     entry_point='tf_rl.envs:GazeboPPOENv',
#     # More arguments here
# )
#
# register(
#     id='GazeboMultiHerEnv-v0',
#     entry_point='tf_rl.envs:GazeboMultiHerEnv',
#     # More arguments here
# )
# register(
#     id='GazeboSingleHerEnv-v0',
#     entry_point='tf_rl.envs:GazeboSingleHerEnv',
#     # More arguments here
# )
# register(
#     id='GazeboMarkerEnv-v0',
#     entry_point='tf_rl.envs:GazeboMarkerEnv',
#     # More arguments here
# )
# register(
#     id='DQNEnv-v0',
#     entry_point='tf_rl.envs:DQNEnv',
#     #20181225添加的dqn环境，改自ddpg多机环境
# )
#
# register(
#     id='MaddpgEnv-v0',
#     entry_point='tf_rl.envs:MaddpgEnv',
# )
#



register(
    id='GazeboMultiJackalLidar-v8',
    entry_point='tf_rl.envs:GazeboMultiJackalLidarEnv8',
    # More arguments here
)
register(
    id='GazeboMultiGoal-v8',
    entry_point='tf_rl.envs:GazeboMultiGoalEnv8',
    # More arguments here
)
register(
    id='SingleDDPGEnv-v0',
    entry_point='tf_rl.envs:SingleDDPGEnv',
)
register(
    id='MaddpgEnv-v0',
    entry_point='tf_rl.envs:MaddpgEnv'
)
register(
    id='DDPGEnv-v0',
    entry_point='tf_rl.envs:DDPGEnv'
)
register(
    id='formDDPGEnv-v0',
    entry_point='tf_rl.envs:formDDPGEnv'
)
