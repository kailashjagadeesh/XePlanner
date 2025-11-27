#include "cbs.h"
#include "utils.h"

#include <limits>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <string>
#include <algorithm>
#include <ctime>

using std::make_shared;
using std::shared_ptr;
using std::vector;

static double heuristicL1(const JointConfig &q, const JointConfig &goal)
{
    double h = 0.0;
    for (size_t i = 0; i < q.size(); ++i)
    {
        h += std::fabs(q[i] - goal[i]);
    }
    return h;
}

static bool violatesVertexConstraints(int agent_id,
                                      const JointConfig &q,
                                      int t,
                                      const std::vector<Constraint> &constraints)
{
    for (const auto &c : constraints)
    {
        if (c.agent_id != agent_id)
            continue;
        if (c.type != ConstraintType::VERTEX)
            continue;
        if (c.t != t)
            continue;

        if (c.q.size() != q.size())
            continue;

        bool same = true;
        for (size_t k = 0; k < q.size(); ++k)
        {
            if (c.q[k] != q[k])
            {
                same = false;
                break;
            }
        }
        if (same)
            return true;
    }
    return false;
}

CBSPlanner::CBSPlanner(mjModel *model,
                       double dq_max,
                       double dt,
                       int num_agents,
                       int dof)
    : model_(model),
      dq_max_(dq_max),
      dt_(dt),
      num_agents_(num_agents),
      dof_(dof)
{
    if (!model_)
    {
        throw std::runtime_error("CBSPlanner: model pointer is null");
    }

    data_plan_ = mj_makeData(model_);
    if (!data_plan_)
    {
        throw std::runtime_error("CBSPlanner: failed to allocate mjData for planning");
    }
}

CBSPlanner::~CBSPlanner()
{
    if (data_plan_)
    {
        mj_deleteData(data_plan_);
        data_plan_ = nullptr;
    }
}

double CBSPlanner::computeTotalCost(const MultiAgentPaths &paths) const
{
    double total = 0.0;
    for (const auto &path : paths)
    {
        total += static_cast<double>(path.size());
    }
    return total;
}

std::vector<Node *> CBSPlanner::buildNodeTrajectory(const MultiAgentPaths &paths) const
{
    size_t T_max = 0;
    for (const auto &path : paths)
    {
        if (path.size() > T_max)
            T_max = path.size();
    }

    std::vector<Node *> traj;
    traj.reserve(T_max);

    double current_time = 0.0;

    // 1. Create the initial node at t=0
    if (T_max > 0)
    {
        std::vector<std::vector<double>> q_multi(num_agents_, std::vector<double>(dof_, 0.0));
        for (int agent = 0; agent < num_agents_; ++agent)
        {
            if (!paths[agent].empty())
                q_multi[agent] = paths[agent][0];
        }
        Node *node = new Node(q_multi, current_time);
        traj.push_back(node);
    }

    // 2. Iterate through steps and calculate duration based on max displacement
    for (size_t t = 0; t < T_max - 1; ++t)
    {
        double max_duration = dt_; // At least one simulation step

        // Find which agent needs the most time for this step
        for (int agent = 0; agent < num_agents_; ++agent)
        {
            const auto &path = paths[agent];
            // If this agent moves in this step (i.e., hasn't finished yet)
            if (t + 1 < path.size())
            {
                const JointConfig &q_curr = path[t];
                const JointConfig &q_next = path[t + 1];

                // Calculate max joint displacement for this agent
                double max_disp = 0.0;
                for (int j = 0; j < dof_; ++j)
                {
                    double d = std::fabs(q_next[j] - q_curr[j]);
                    if (d > max_disp)
                        max_disp = d;
                }

                // Time required = distance / velocity
                double duration = max_disp / dq_max_;
                if (duration > max_duration)
                    max_duration = duration;
            }
        }

        // Increment time
        current_time += max_duration;

        // Create the node for the next step
        std::vector<std::vector<double>> q_multi(num_agents_, std::vector<double>(dof_, 0.0));
        for (int agent = 0; agent < num_agents_; ++agent)
        {
            const auto &path = paths[agent];
            // If agent path finished, keep using the last config
            const JointConfig &q = (t + 1 < path.size()) ? path[t + 1] : path.back();
            q_multi[agent] = q;
        }
        Node *node = new Node(q_multi, current_time);
        traj.push_back(node);
    }

    return traj;
}

std::vector<Constraint> CBSPlanner::getConstraintsForAgent(int agent_id, const std::vector<Constraint> &all_constraints) const
{
    std::vector<Constraint> result;
    for (const auto &c : all_constraints)
    {
        if (c.agent_id == agent_id)
        {
            result.push_back(c);
        }
    }
    return result;
}

bool CBSPlanner::findFirstConflict(const MultiAgentPaths &paths, Conflict &out_conflict)
{
    size_t T_max = 0;
    for (const auto &path : paths)
    {
        if (path.size() > T_max)
            T_max = path.size();
    }

    if (T_max == 0)
        return false;

    for (size_t t = 0; t < T_max; ++t)
    {
        std::vector<std::vector<double>> q_multi(num_agents_,
                                                 std::vector<double>(dof_, 0.0));

        for (int agent = 0; agent < num_agents_; ++agent)
        {
            const auto &path = paths[agent];
            const JointConfig &q = (t < path.size()) ? path[t] : path.back();
            q_multi[agent] = q;
        }

        setAllArmsQpos(model_, data_plan_, q_multi);
        mju_zero(data_plan_->qvel, model_->nv);
        mju_zero(data_plan_->qacc, model_->nv);
        mju_zero(data_plan_->ctrl, model_->nu);

        mj_forward(model_, data_plan_);

        for (int ci = 0; ci < data_plan_->ncon; ++ci)
        {
            const mjContact &c = data_plan_->contact[ci];
            if (c.dist >= 0.0)
                continue;

            int a1 = geomToAgent(model_, c.geom1, num_agents_);
            int a2 = geomToAgent(model_, c.geom2, num_agents_);

            if (a1 >= 0 && a2 >= 0 && a1 != a2)
            {
                out_conflict.agent1 = a1;
                out_conflict.agent2 = a2;
                out_conflict.t = static_cast<int>(t);
                out_conflict.is_edge_conflict = false;

                std::cout << "[CBS] Conflict between agent " << a1
                          << " and agent " << a2
                          << " at time " << t << std::endl;
                return true;
            }
        }
    }

    return false;
}

void CBSPlanner::expandNode(const std::shared_ptr<CTNode> &node,
                            const Conflict &conflict,
                            std::priority_queue<std::shared_ptr<CTNode>,
                                                std::vector<std::shared_ptr<CTNode>>,
                                                CTNodeCompare> &open)
{
    int i = conflict.agent1;
    int j = conflict.agent2;
    int t = conflict.t;

    auto make_child = [&](int agent_to_constrain) -> std::shared_ptr<CTNode>
    {
        auto child = std::make_shared<CTNode>(*node);

        Constraint c;
        c.agent_id = agent_to_constrain;
        c.type = ConstraintType::VERTEX;
        c.t = t;

        const AgentPath &path = node->paths[agent_to_constrain];
        const JointConfig &q_at_t = (t < (int)path.size()) ? path[t] : path.back();
        c.q = q_at_t;

        child->constraints.push_back(c);

        AgentPath new_path;
        bool ok = lowLevelPlan(agent_to_constrain,
                               start_configs_[agent_to_constrain],
                               goal_configs_[agent_to_constrain],
                               child->constraints,
                               new_path);
        if (!ok)
        {
            return nullptr;
        }

        child->paths[agent_to_constrain] = std::move(new_path);
        child->cost = computeTotalCost(child->paths);
        child->depth = node->depth + 1;

        return child;
    };

    auto child_i = make_child(i);
    if (child_i)
    {
        open.push(child_i);
    }

    auto child_j = make_child(j);
    if (child_j)
    {
        open.push(child_j);
    }
}

// bool CBSPlanner::lowLevelPlan(int agent_id,
//                               const JointConfig &q_start,
//                               const JointConfig &q_goal,
//                               const std::vector<Constraint> &all_constraints,
//                               AgentPath &out_path)
// {
//     // Get constraints relevant to this agent
//     std::vector<Constraint> constraints = getConstraintsForAgent(agent_id, all_constraints);

//     struct LLNode
//     {
//         JointConfig q;
//         int t;
//         double g;
//         double f;
//         int parent_idx;
//     };

//     const double w_factor = 5.0;
//     const double step_factor = 3.0;
//     const double step = dq_max_ * dt_ * step_factor; // joint step size per discrete step
//     const int max_steps = 15000;                      // search horizon (tune later)
//     const double goal_tol = 0.05;                    // joint-space goal tolerance

//     std::vector<LLNode> nodes;
//     nodes.reserve(1024);

//     struct PQItem
//     {
//         double f;
//         int idx;
//         bool operator<(const PQItem &o) const { return f > o.f; }
//     };

//     std::priority_queue<PQItem> open;

//     // hash for visited states with best g so far
//     std::unordered_map<std::string, double> best_g;

//     long long iteration_count = 0;

//     auto keyFromQ = [&](const JointConfig &q) -> std::string
//     {
//         std::string key;
//         key.reserve(q.size() * 8);
//         for (double v : q)
//         {
//             int iv = static_cast<int>(std::round(v / (step * 0.5)));
//             key.append(std::to_string(iv));
//             key.push_back('_');
//         }
//         return key;
//     };

//     LLNode start;
//     start.q = q_start;
//     start.t = 0;
//     start.g = 0.0;
//     start.f = start.g + w_factor * heuristicL1(q_start, q_goal);
//     start.parent_idx = -1;

//     int start_idx = (int)nodes.size();
//     nodes.push_back(start);
//     open.push({start.f, start_idx});
//     best_g[keyFromQ(start.q)] = 0.0;

//     // ----- A* loop -----
//     while (!open.empty())
//     {
//         iteration_count++;
//         if (iteration_count % 10000 == 0)
//         {
//             std::cout << "[A*] Agent " << agent_id << " Iterations: " << iteration_count
//                       << ", Open Size: " << open.size()
//                       << ", Nodes Explored: " << nodes.size() << std::endl;
//         }
//         auto [f_curr, idx] = open.top();
//         open.pop();

//         LLNode &curr = nodes[idx];

//         // goal check
//         if (heuristicL1(curr.q, q_goal) < goal_tol)
//         {
//             // reconstruct path
//             AgentPath path;
//             int cur = idx;
//             while (cur >= 0)
//             {
//                 path.push_back(nodes[cur].q);
//                 cur = nodes[cur].parent_idx;
//             }
//             std::reverse(path.begin(), path.end());
//             out_path = std::move(path);
//             return true;
//         }

//         if (curr.t >= max_steps)
//             continue;

//         std::vector<JointConfig> neighbors;

//         // 1) WAIT action: stay still
//         neighbors.push_back(curr.q);

//         // 2) single-joint +/- step moves
//         for (int j = 0; j < dof_; ++j)
//         {
//             JointConfig q_plus = curr.q;
//             JointConfig q_minus = curr.q;

//             q_plus[j] += step;
//             q_minus[j] -= step;

//             neighbors.push_back(q_plus);
//             neighbors.push_back(q_minus);
//         }

//         for (const auto &q_next : neighbors)
//         {
//             int t_next = curr.t + 1;

//             // check CBS vertex constraints
//             if (violatesVertexConstraints(agent_id, q_next, t_next, constraints))
//                 continue;

//             // collision check: this agent moves, others held at their start configs
//             std::vector<std::vector<double>> q_multi(num_agents_, std::vector<double>(dof_, 0.0));
//             for (int a = 0; a < num_agents_; ++a)
//             {
//                 if (a == agent_id)
//                     q_multi[a] = q_next;
//                 else
//                     q_multi[a] = start_configs_[a]; // keep other arms fixed (env + self collisions only)
//             }

//             if (!isStateValid(model_, data_plan_, q_multi, false))
//                 continue;

//             double g_next = curr.g + 1.0;
//             std::string key = keyFromQ(q_next);

//             auto it = best_g.find(key);
//             if (it != best_g.end() && g_next >= it->second)
//             {
//                 continue;
//             }
//             best_g[key] = g_next;

//             LLNode succ;
//             succ.q = q_next;
//             succ.t = t_next;
//             succ.g = g_next;
//             succ.f = g_next + w_factor * heuristicL1(q_next, q_goal);
//             succ.parent_idx = idx;

//             int succ_idx = (int)nodes.size();
//             nodes.push_back(std::move(succ));
//             open.push({nodes[succ_idx].f, succ_idx});
//         }
//     }

//     std::cerr << "[CBS] lowLevelPlan failed for agent " << agent_id << std::endl;
//     return false;
// }

// RRT-Connect Implementation for lowLevelPlan
bool CBSPlanner::lowLevelPlan(int agent_id,
                              const JointConfig &q_start,
                              const JointConfig &q_goal,
                              const std::vector<Constraint> &all_constraints,
                              AgentPath &out_path)
{
    std::srand(std::time(0));

    auto distanceL1 = [&](const JointConfig &q1, const JointConfig &q2) -> double
    {
        double d = 0.0;
        for (size_t i = 0; i < q1.size(); ++i)
        {
            d += std::fabs(q1[i] - q2[i]);
        }
        return d;
    };

    auto sampleRandomQ = [&](int dof) -> JointConfig
    {
        JointConfig q(dof);
        const double max_range = M_PI;
        for (int i = 0; i < dof; ++i)
        {
            double r = static_cast<double>(rand()) / RAND_MAX;
            q[i] = -max_range + r * (2.0 * max_range);
        }
        return q;
    };

    auto steer = [&](const JointConfig &q_from, const JointConfig &q_to, double max_step, int dof) -> JointConfig
    {
        double dist = distanceL1(q_from, q_to);
        if (dist < max_step)
            return q_to;

        JointConfig q_new = q_from;
        for (int i = 0; i < dof; ++i)
        {
            double dir = q_to[i] - q_from[i];
            q_new[i] += dir / dist * max_step;
        }
        return q_new;
    };

    struct LLNode
    {
        JointConfig q;
        int parent_idx;
    };

    const double RRT_STEP_SIZE = dq_max_ * dt_ * 5.0; // Step factor 5.0
    const int MAX_RRT_NODES = 50000;                  // Increased node limit
    const double GOAL_TOL = 0.01;                     // Tightened goal tolerance
    const double GOAL_BIAS_PROB = 0.15;               // Goal sampling probability
    const int NUM_EDGE_SUBSTEPS = 5;                  // Discretization for edge check
    const std::vector<double> safe_q(dof_, 0.0);      // Safe pose for non-moving arms (used for collision check)

    std::vector<LLNode> tree_a;
    std::vector<LLNode> tree_b;
    tree_a.reserve(MAX_RRT_NODES);
    tree_b.reserve(MAX_RRT_NODES);

    LLNode start_node = {q_start, -1};
    LLNode goal_node = {q_goal, -1};
    tree_a.push_back(start_node);
    tree_b.push_back(goal_node);

    auto findNearest = [&](const std::vector<LLNode> &tree, const JointConfig &q_target) -> int
    {
        int q_near_idx = -1;
        double min_dist = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < tree.size(); ++i)
        {
            double dist = distanceL1(tree[i].q, q_target);
            if (dist < min_dist)
            {
                min_dist = dist;
                q_near_idx = static_cast<int>(i);
            }
        }
        return q_near_idx;
    };

    auto extend = [&](std::vector<LLNode> &tree, const JointConfig &q_target, int &out_new_idx) -> bool
    {
        int q_near_idx = findNearest(tree, q_target);
        const JointConfig &q_near = tree[q_near_idx].q;
        JointConfig q_new = steer(q_near, q_target, RRT_STEP_SIZE, dof_);

        std::vector<std::vector<double>> q_from(num_agents_, safe_q);
        std::vector<std::vector<double>> q_to(num_agents_, safe_q);

        q_from[agent_id] = q_near;
        q_to[agent_id] = q_new;

        if (!isEdgeValid(model_, data_plan_, q_from, q_to, NUM_EDGE_SUBSTEPS, false))
            return false;

        if (violatesVertexConstraints(agent_id, q_new, tree.size(), getConstraintsForAgent(agent_id, all_constraints)))
            return false;

        LLNode new_node = {q_new, q_near_idx};
        out_new_idx = static_cast<int>(tree.size());
        tree.push_back(new_node);
        return true;
    };

    auto connect = [&](std::vector<LLNode> &tree_to, const JointConfig &q_target, int &out_last_idx) -> bool
    {
        int q_new_idx = -1;
        while (distanceL1(tree_to.back().q, q_target) > RRT_STEP_SIZE)
        {
            if (!extend(tree_to, q_target, q_new_idx))
                return false;
        }

        out_last_idx = findNearest(tree_to, q_target);
        if (distanceL1(tree_to[out_last_idx].q, q_target) < GOAL_TOL)
        {
            return true;
        }

        return false;
    };

    for (int k = 0; k < MAX_RRT_NODES; ++k)
    {
        int q_b_last_idx = -1;

        JointConfig q_rand;
        if (static_cast<double>(rand()) / RAND_MAX < GOAL_BIAS_PROB)
        {
            q_rand = q_goal;
        }
        else
        {
            q_rand = sampleRandomQ(dof_);
        }

        int q_a_new_idx = -1;
        if (!extend(tree_a, q_rand, q_a_new_idx))
        {
            goto swap_trees;
        }

        if (connect(tree_b, tree_a[q_a_new_idx].q, q_b_last_idx))
        {
            AgentPath path_a;
            int cur_a = q_a_new_idx;
            while (cur_a != -1)
            {
                path_a.push_back(tree_a[cur_a].q);
                cur_a = tree_a[cur_a].parent_idx;
            }
            std::reverse(path_a.begin(), path_a.end());

            AgentPath path_b;
            int cur_b = q_b_last_idx;
            while (cur_b != -1)
            {
                path_b.push_back(tree_b[cur_b].q);
                cur_b = tree_b[cur_b].parent_idx;
            }

            out_path.insert(out_path.end(), path_a.begin(), path_a.end());
            out_path.insert(out_path.end(), path_b.begin(), path_b.end());

            std::cout << "[RRT-Connect] Agent " << agent_id << " found path in " << k + 1 << " iterations. Total nodes: " << tree_a.size() + tree_b.size() << ".\n";
            return true;
        }

    swap_trees:
        std::vector<LLNode> *ptr_a = &tree_a;
        std::vector<LLNode> *ptr_b = &tree_b;

        std::swap(ptr_a, ptr_b);
    }

    std::cerr << "[CBS] lowLevelPlan failed (RRT-Connect exhausted) for agent " << agent_id << std::endl;
    return false;
}

bool CBSPlanner::debugLowLevelPlan(int agent_id,
                                   const JointConfig &q_start,
                                   const JointConfig &q_goal,
                                   const std::vector<Constraint> &constraints,
                                   AgentPath &out_path)
{
    start_configs_.assign(num_agents_, std::vector<double>(dof_, 0.0));
    start_configs_[agent_id] = q_start;

    return lowLevelPlan(agent_id, q_start, q_goal, constraints, out_path);
}

// High-level CBS loop
std::vector<Node *> CBSPlanner::plan(const std::vector<std::vector<double>> &start_poses,
                                     const std::vector<std::vector<double>> &goal_poses)
{
    if ((int)start_poses.size() != num_agents_ ||
        (int)goal_poses.size() != num_agents_)
    {
        std::cerr << "[CBS] Error: start_poses / goal_poses size mismatch with num_agents\n";
        return {};
    }
    for (int i = 0; i < num_agents_; ++i)
    {
        if ((int)start_poses[i].size() != dof_ ||
            (int)goal_poses[i].size() != dof_)
        {
            std::cerr << "[CBS] Error: DOF mismatch for agent " << i << "\n";
            return {};
        }
    }

    start_configs_.assign(start_poses.begin(), start_poses.end());
    goal_configs_.assign(goal_poses.begin(), goal_poses.end());

    auto root = make_shared<CTNode>();
    root->constraints.clear();
    root->paths.resize(num_agents_);

    for (int agent = 0; agent < num_agents_; ++agent)
    {
        JointConfig q_start = start_poses[agent];
        JointConfig q_goal = goal_poses[agent];

        std::cout << "[CBS] Starting Low-Level Plan for Agent " << agent
                  << " (T=0, Cost=0) from start pose..." << std::endl;

        AgentPath path_i;
        bool ok = lowLevelPlan(agent, q_start, q_goal, root->constraints, path_i);
        if (!ok)
        {
            std::cerr << "[CBS] Root low-level plan failed for agent " << agent << "\n";
            return {};
        }
        root->paths[agent] = std::move(path_i);
    }

    root->cost = computeTotalCost(root->paths);
    root->depth = 0;

    std::priority_queue<shared_ptr<CTNode>, std::vector<shared_ptr<CTNode>>, CTNodeCompare> open;
    open.push(root);

    while (!open.empty())
    {
        auto node = open.top();
        open.pop();

        Conflict conflict;
        bool has_conflict = findFirstConflict(node->paths, conflict);

        if (!has_conflict)
        {
            std::cout << "[CBS] Found conflict-free solution with cost " << node->cost << "\n";
            return buildNodeTrajectory(node->paths);
        }

        expandNode(node, conflict, open);
    }

    std::cerr << "[CBS] No solution found (OPEN exhausted)\n";
    return {};
}
