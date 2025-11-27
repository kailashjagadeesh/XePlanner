#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <cmath>
#include <memory>
#include "utils.h"

// A simple node structure for the RRT tree
struct RRTNode
{
    std::vector<double> q;
    int parent_index;
    
    RRTNode(const std::vector<double>& _q, int _parent) : q(_q), parent_index(_parent) {}
};

class RRTConnectPlanner
{
public:
    RRTConnectPlanner(mjModel *model, int agent_id, int dof, int num_agents);
    ~RRTConnectPlanner();

    // Main planning function
    std::vector<Node *> plan(const std::vector<double> &start_conf, const std::vector<double> &goal_conf);

private:
    // RRT Helper functions
    double distance(const std::vector<double> &q1, const std::vector<double> &q2);
    std::vector<double> sample(const std::vector<double>& goal);
    std::vector<double> steer(const std::vector<double> &from, const std::vector<double> &to);
    
    // Core RRT primitives
    // Returns index of new node if successful, -1 if failed/collision
    int extend(std::vector<RRTNode> &tree, const std::vector<double> &q_target);
    
    // Returns true if connection made to q_target, false otherwise
    // Updates connection_idx with the index in the tree that connected
    bool connect(std::vector<RRTNode> &tree, const std::vector<double> &q_target, int &connection_idx);

    // Checks
    bool isValid(const std::vector<double>& q);
    bool isEdgeFree(const std::vector<double>& q_start, const std::vector<double>& q_end);

private:
    mjModel *model_;
    mjData *data_; // Local data for planning collision checks
    
    int agent_id_;
    int dof_;
    int num_agents_;
    
    // Parameters
    double step_size_;
    int max_nodes_;
    double goal_bias_;
    double goal_tolerance_;
};