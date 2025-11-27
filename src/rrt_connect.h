#pragma once

#include <vector>
#include <mujoco/mujoco.h>
#include "utils.h"

// Simple RRT-Connect planner that returns a sequence of joint-space waypoints.
// It expects joint positions in Node::q shaped as q[arm][joint].
std::vector<Node*> rrtConnect(
    mjModel* model,
    int num_actuators,
    int dof,
    const Node* start,
    const Node* goal,
    int max_iters,
    double step_size);
