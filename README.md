# DRL_Nav
Robot Formation Navigation in Gazebo with Deep Reinforcement Learning.

---

## DDPG Formation Navigation：
The _DDPG_formation_test.py_ and _DDPG_formation_train.py_ are files for multi robots navigation. However, they still be idiot.
- The robots4_formation.launch and robots8_formation.launch are for my previous tests.
- The multi_goal.launch belongs to the predecessor. The corresponding training and testing scripts are saved on the disk.

---
## System Setup:

### Add New Env:
Register the class in the tf_rl/\_\_init__.py and import the env script in the  tf_rl/envs/\_\_init__.py.

### World Files Introduction:
The main world files used in our project are described below. If you want to modify the world file, you should 
- **ddpg3Block.world**: 3 cinder blocks lying on the points (-2,-2) (0,0) (2,2). The rectangle wall from (-6,-9) to (8.5,6).
- **maddpg1block.world**: 1 jersey_barrier lying on the middle. Approximatly from (-0.3,0) to (3,-4). The rectangle wall from (-6,-9) to (8.5,6).
- **maddpg111.world**: 3 jersey_barrier lying in parallel. From (-6,-4) to (2.2,-7), (0,0) to (8.5,-1.3), (-6,2) to (2.1,3). The rectangle wall from (-6,-9) to (8.5,6).
- **maddpgFree.world**： Free environment. The rectangle wall from (-6,-9) to (8.5,6).
- **maddpgblock.world** and **wallblock.world** need to be evaluated later.
