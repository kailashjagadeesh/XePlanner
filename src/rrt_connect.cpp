#include "rrt_connect.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <limits>

using namespace std;

RRTConnectPlanner::RRTConnectPlanner(mjModel *model, int agent_id, int dof, int num_agents)
    : model_(model), agent_id_(agent_id), dof_(dof), num_agents_(num_agents)
{
    // Create a separate mjData for planning to avoid thread conflicts with visualization
    data_ = mj_makeData(model);
    
    // ROBUST PARAMETERS
    step_size_ = 0.5;      // Large step size for faster exploration
    max_nodes_ = 50000;    // High node limit
    goal_bias_ = 0.10;     // 10% chance to sample goal
    goal_tolerance_ = 0.05;
}

RRTConnectPlanner::~RRTConnectPlanner()
{
    if (data_)
        mj_deleteData(data_);
}

double RRTConnectPlanner::distance(const vector<double> &q1, const vector<double> &q2)
{
    double d = 0.0;
    for (int i = 0; i < dof_; ++i)
        d += fabs(q1[i] - q2[i]);
    return d;
}

vector<double> RRTConnectPlanner::sample(const vector<double>& goal)
{
    vector<double> q(dof_);
    
    // Goal biasing
    if ((double)rand() / RAND_MAX < goal_bias_)
    {
        return goal;
    }

    // Uniform sampling [-pi, pi]
    for (int i = 0; i < dof_; ++i)
    {
        double r = (double)rand() / RAND_MAX;
        q[i] = -3.14159 + r * (2.0 * 3.14159);
    }
    return q;
}

vector<double> RRTConnectPlanner::steer(const vector<double> &from, const vector<double> &to)
{
    double d = distance(from, to);
    if (d < step_size_)
        return to;

    vector<double> new_q(dof_);
    for (int i = 0; i < dof_; ++i)
    {
        double dir = to[i] - from[i];
        new_q[i] = from[i] + (dir / d) * step_size_;
    }
    return new_q;
}

bool RRTConnectPlanner::isValid(const vector<double>& q)
{
    // Build multi-agent config: Target agent gets 'q', others get 0.0 (Safe Pose)
    vector<vector<double>> q_multi(num_agents_, vector<double>(dof_, 0.0));
    q_multi[agent_id_] = q;

    return isStateValid(model_, data_, q_multi, false);
}

bool RRTConnectPlanner::isEdgeFree(const vector<double>& q_start, const vector<double>& q_end)
{
    // Build multi-agent config
    vector<vector<double>> q_multi_start(num_agents_, vector<double>(dof_, 0.0));
    vector<vector<double>> q_multi_end(num_agents_, vector<double>(dof_, 0.0));
    
    q_multi_start[agent_id_] = q_start;
    q_multi_end[agent_id_] = q_end;

    // Check 5 sub-steps along the edge
    return isEdgeValid(model_, data_, q_multi_start, q_multi_end, 5, false);
}

int RRTConnectPlanner::extend(vector<RRTNode> &tree, const vector<double> &q_target)
{
    // 1. Find Nearest
    int nearest_idx = -1;
    double min_dist = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < tree.size(); ++i)
    {
        double d = distance(tree[i].q, q_target);
        if (d < min_dist)
        {
            min_dist = d;
            nearest_idx = i;
        }
    }

    // 2. Steer
    vector<double> q_new = steer(tree[nearest_idx].q, q_target);

    // 3. Check Collision (Node + Edge)
    if (!isValid(q_new)) return -1;
    if (!isEdgeFree(tree[nearest_idx].q, q_new)) return -1;

    // 4. Add to Tree
    tree.emplace_back(q_new, nearest_idx);
    return tree.size() - 1;
}

bool RRTConnectPlanner::connect(vector<RRTNode> &tree, const vector<double> &q_target, int &connection_idx)
{
    vector<double> q_curr = q_target;
    
    while (true)
    {
        int new_idx = extend(tree, q_curr);
        
        if (new_idx == -1) 
        {
            // Extension failed (collision)
            return false; 
        }

        // Check if we reached the target configuration
        if (distance(tree[new_idx].q, q_target) < 1e-3) // Close enough to specific target point
        {
            connection_idx = new_idx;
            return true;
        }
        
        // Update current target to be the node we just added (though steer handles direction)
        // Actually, extend takes a target and steers towards it. 
        // To "Connect", we keep extending towards the *original* target.
        // If we reached the target via steer, extend returns the node at q_target.
        
        // Simply: if extend succeeded, check if we are at the goal
        if (distance(tree[new_idx].q, q_target) < step_size_)
        {
             // One more step might land us there, or we are close enough
             // Let's force a final check/add if needed, or accept tolerance
             connection_idx = new_idx;
             return true;
        }
    }
}

std::vector<Node *> RRTConnectPlanner::plan(const std::vector<double> &start_conf, const std::vector<double> &goal_conf)
{
    srand(time(0));
    
    if (!isValid(start_conf) || !isValid(goal_conf))
    {
        cout << "[RRT] Error: Start or Goal is invalid!" << endl;
        return {};
    }

    // Tree A (Start), Tree B (Goal)
    vector<RRTNode> tree_a; 
    vector<RRTNode> tree_b;
    tree_a.reserve(max_nodes_);
    tree_b.reserve(max_nodes_);

    tree_a.emplace_back(start_conf, -1);
    tree_b.emplace_back(goal_conf, -1);

    cout << "[RRT] Planning started..." << endl;

    for (int k = 0; k < max_nodes_; ++k)
    {
        // 1. Sample
        vector<double> q_rand = sample(goal_conf); // Passing goal just for bias reference if needed

        // 2. Extend A
        int idx_a = extend(tree_a, q_rand);

        if (idx_a != -1) // If A extended successfully
        {
            // 3. Connect B to A's new node
            int idx_b = -1;
            // Try to connect Tree B to the specific configuration added to Tree A
            if (connect(tree_b, tree_a[idx_a].q, idx_b))
            {
                cout << "[RRT] Path found at iter " << k << "! Nodes: " << tree_a.size() + tree_b.size() << endl;

                // Reconstruct Path
                // Path A: Start -> ... -> idx_a
                vector<vector<double>> full_path_q;
                
                int curr = idx_a;
                while(curr != -1) {
                    full_path_q.push_back(tree_a[curr].q);
                    curr = tree_a[curr].parent_index;
                }
                std::reverse(full_path_q.begin(), full_path_q.end());

                // Path B: Goal -> ... -> idx_b (connect point)
                curr = idx_b;
                while(curr != -1) {
                    full_path_q.push_back(tree_b[curr].q);
                    curr = tree_b[curr].parent_index;
                }

                // Convert to Node* format for main.cpp
                vector<Node*> result;
                for(const auto& q : full_path_q)
                {
                    // Convert single-agent config to multi-agent config (others at 0)
                    vector<vector<double>> q_multi(num_agents_, vector<double>(dof_, 0.0));
                    q_multi[agent_id_] = q;
                    // Time is dummy here, densifyPlan handles it
                    result.push_back(new Node(q_multi, 0.0));
                }
                return result;
            }
        }

        // 4. Swap Trees
        std::swap(tree_a, tree_b);
    }

    cout << "[RRT] Failed to find path." << endl;
    return {};
}