# Simple-PPO
Simple PPO for pushing an object

To-Do:
1. Gym type PyBullet environment for push-to-edge problem
2. Trainer script to run stable baselines3 PPO on this environment
3. Try adding Hindsight Experience Replay of Stable baselines3

## Setup Instructions
1. Download the Assets zip from this link - https://drive.google.com/file/d/1W6cfUSt1Og0hJplirAs9SPudb8Nlfjh4/view?usp=sharing 
2. Unzip the Assets in the current directory (git repo) (refer the directory structure in the next section to get a better idea)
3. Install the environment from environment.yml using conda
4. Activate the environment
5. Run ppo.py

## Directory Structure
.
├── Assets
│   ├── blocks
│   │   ├── concavebackup.obj
│   │   ├── concave.obj
│   │   ├── concave.urdf
│   │   ├── cube.obj
│   │   ├── cube.urdf
│   │   ├── cylinder.obj
│   │   ├── cylinder.urdf
│   │   ├── half-cube.obj
│   │   ├── half-cube.urdf
│   │   ├── half-cylinder.obj
│   │   ├── half-cylinder.urdf
│   │   ├── potato_chip_1_v1.urdf
│   │   ├── rect.obj
│   │   ├── rect.urdf
│   │   ├── triangle.obj
│   │   └── triangle.urdf
│   ├── ur5e
│   │   ├── collision
│   │   │   ├── base.stl
│   │   │   ├── forearm.stl
│   │   │   ├── shoulder.stl
│   │   │   ├── upperarm.stl
│   │   │   ├── wrist1.stl
│   │   │   ├── wrist2.stl
│   │   │   └── wrist3.stl
│   │   ├── gripper
│   │   │   ├── collision
│   │   │   │   ├── robotiq_arg2f_85_base_link.stl
│   │   │   │   ├── robotiq_arg2f_85_inner_finger.dae
│   │   │   │   ├── robotiq_arg2f_85_inner_knuckle.dae
│   │   │   │   ├── robotiq_arg2f_85_outer_finger.dae
│   │   │   │   ├── robotiq_arg2f_85_outer_knuckle.dae
│   │   │   │   └── robotiq_arg2f_base_link.stl
│   │   │   ├── robotiq_2f_85.urdf
│   │   │   ├── textures
│   │   │   │   ├── gripper-2f_BaseColor.jpg
│   │   │   │   ├── gripper-2f_Metallic.jpg
│   │   │   │   ├── gripper-2f_Normal.jpg
│   │   │   │   └── gripper-2f_Roughness.jpg
│   │   │   └── visual
│   │   │       ├── robotiq_arg2f_85_base_link.dae
│   │   │       ├── robotiq_arg2f_85_inner_finger.dae
│   │   │       ├── robotiq_arg2f_85_inner_knuckle.dae
│   │   │       ├── robotiq_arg2f_85_outer_finger.dae
│   │   │       ├── robotiq_arg2f_85_outer_knuckle.dae
│   │   │       ├── robotiq_arg2f_85_pad.dae
│   │   │       └── robotiq_gripper_coupling.stl
│   │   ├── license.txt
│   │   ├── plane.obj
│   │   ├── spatula
│   │   │   ├── base.obj
│   │   │   ├── spatula-base.urdf
│   │   │   ├── tip3.mtl
│   │   │   ├── tip3.obj
│   │   │   ├── tip_flat.mtl
│   │   │   ├── tip_flat.obj
│   │   │   ├── tip.mtl
│   │   │   └── tip.obj
│   │   ├── ur5e.urdf
│   │   └── visual
│   │       ├── base.dae
│   │       ├── base.mtl
│   │       ├── base.obj
│   │       ├── forearm.dae
│   │       ├── forearm.mtl
│   │       ├── forearm.obj
│   │       ├── shoulder.dae
│   │       ├── shoulder.mtl
│   │       ├── shoulder.obj
│   │       ├── upperarm.dae
│   │       ├── upperarm.mtl
│   │       ├── upperarm.obj
│   │       ├── wrist1.dae
│   │       ├── wrist1.mtl
│   │       ├── wrist1.obj
│   │       ├── wrist2.dae
│   │       ├── wrist2.mtl
│   │       ├── wrist2.obj
│   │       ├── wrist3.dae
│   │       ├── wrist3.mtl
│   │       └── wrist3.obj
│   └── workspace
│       ├── bamboo_albedo.jpg
│       ├── bamboo_rpoughness.png
│       ├── bottom.mtl
│       ├── bottom.obj
│       ├── bottom.urdf
│       ├── None_normal.png
│       ├── plane.obj
│       ├── wall.urdf
│       ├── wallx30.mtl
│       ├── wallx30.obj
│       ├── wally32.mtl
│       ├── wally32.obj
│       └── workspace.urdf
├── Config
│   ├── constants.py
│   └── __pycache__
│       └── constants.cpython-38.pyc
├── Environments
│   ├── cameras.py
│   ├── environment_sim.py
│   ├── __pycache__
│   │   ├── cameras.cpython-38.pyc
│   │   ├── environment_sim.cpython-38.pyc
│   │   ├── single_object_case.cpython-38.pyc
│   │   └── utils.cpython-38.pyc
│   ├── single_object_case.py
│   └── utils.py
├── ppo.py
├── pushGymEnv.py
├── __pycache__
│   └── pushGymEnv.cpython-38.pyc
├── README.md
├── Results
└── Utils
    ├── actionUtils.py
    ├── __pycache__
    │   ├── actionUtils.cpython-38.pyc
    │   └── rewardUtils.cpython-38.pyc
    ├── rewardUtils.py
    └── terminationUtils.py