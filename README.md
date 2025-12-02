This repo is for the course project for 16-782: Planning and Decision Making in Robotics, course by Maxim Likhachev @ Carnegie Mellon University during the fall semester of 2025.
The project is to experiment with planners for multi-manipulator actions.


Install (Tested on Ubuntu 22.04):
```
wget https://github.com/google-deepmind/mujoco/releases/download/3.3.7/mujoco-3.3.7-linux-x86_64.tar.gz
tar -xvf mujoco-3.3.7-linux-x86_64.tar.gz
sudo mv mujoco-3.3.7 /usr/local/mujoco 
echo "/usr/local/mujoco/lib" | sudo tee /etc/ld.so.conf.d/mujoco.conf
sudo ldconfig

sudo apt install libglfw3 libglfw3-dev
```

To run the planners, build the workspace and run
``` ./CBSPlanner ``` for CBS planner
``` ./ECBSPlanner ``` for ECBS planner
``` ./RRTPlanner ``` for RRT-Connect planner

Random start/goal generator (collision-checked):
- Build target: `cmake -S . -B build && cmake --build build --target RandomConfigGenerator`
- Run from `build/`: `./RandomConfigGenerator [num_pairs] [output_path] [seed]`
- Output: JSON with four 7-DOF joint arrays under `"start"` and `"goal"` for each sample (default `random_configs.json` in the run directory).

2 new configs added:
- multi_panda_criss_cross.xml
- panda_four_bins.xml

add these names in scene.xml. The start config for multi_panda_criss_cross.xml is in main_cbs.cpp

Franka Panda Model Adapated From:
https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
