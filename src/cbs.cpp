#include "cbs.h"
#include "rrt_connect.h"
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
    const double EPSILON = 0.1; // Tolerance in radians. RRT is noisy.

    for (const auto &c : constraints)
    {
        // Skip if its for a different agent, different time step or not a vertex constraint
        if (c.agent_id != agent_id)
            continue;
        if (c.type != ConstraintType::VERTEX)
            continue;
        if (c.t != t)
            continue;

        if (c.q.size() != q.size())
            continue;

        // Check if q is "close enough" to the constraint c.q
        bool within_collision_radius = true;
        for (size_t k = 0; k < q.size(); ++k)
        {
            if (std::fabs(c.q[k] - q[k]) > EPSILON)
            {
                within_collision_radius = false;
                break;
            }
        }

        // If we are close to the constraint, it is a violation
        if (within_collision_radius)
            return true;
    }
    return false;
}

CBSPlanner::CBSPlanner(mjModel *model, double dq_max, double dt, int num_agents, int dof)
    : model_(model), dq_max_(dq_max), dt_(dt), num_agents_(num_agents), dof_(dof)
{
    if (!model_)
        throw std::runtime_error("CBSPlanner: model pointer is null");
    data_plan_ = mj_makeData(model_);
    if (!data_plan_)
        throw std::runtime_error("CBSPlanner: failed to allocate mjData for planning");
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
        total += static_cast<double>(path.size());
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

    for (size_t t = 0; t < T_max - 1; ++t)
    {
        double max_duration = dt_;

        for (int agent = 0; agent < num_agents_; ++agent)
        {
            const auto &path = paths[agent];
            if (t + 1 < path.size())
            {
                const JointConfig &q_curr = path[t];
                const JointConfig &q_next = path[t + 1];
                double max_disp = 0.0;
                for (int j = 0; j < dof_; ++j)
                {
                    double d = std::fabs(q_next[j] - q_curr[j]);
                    if (d > max_disp)
                        max_disp = d;
                }
                double duration = max_disp / dq_max_;
                if (duration > max_duration)
                    max_duration = duration;
            }
        }
        current_time += max_duration;

        std::vector<std::vector<double>> q_multi(num_agents_, std::vector<double>(dof_, 0.0));
        for (int agent = 0; agent < num_agents_; ++agent)
        {
            const auto &path = paths[agent];
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
            result.push_back(c);
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

    // Match the simulation resolution ---
    // In main.cpp, densifyPlan uses dt=0.01. Waypoints are t=0,1,2...
    // So there are 100 interpolation steps per waypoint interval.
    // We must check ALL of them to ensure no collisions in blind spots.
    const int INTERPOLATION_STEPS = 100;
    const double SAFETY_MARGIN = 0.02; // 2cm extra buffer

    for (size_t t = 0; t < T_max; ++t)
    {

        int steps_to_check = (t == T_max - 1) ? 1 : INTERPOLATION_STEPS;

        for (int k = 0; k < steps_to_check; ++k)
        {
            double alpha = static_cast<double>(k) / static_cast<double>(INTERPOLATION_STEPS);

            std::vector<std::vector<double>> q_multi(num_agents_, std::vector<double>(dof_, 0.0));

            for (int agent = 0; agent < num_agents_; ++agent)
            {
                const auto &path = paths[agent];
                const JointConfig &q_curr = (t < path.size()) ? path[t] : path.back();
                const JointConfig &q_next = (t + 1 < path.size()) ? path[t + 1] : path.back();

                for (int j = 0; j < dof_; ++j)
                {
                    q_multi[agent][j] = (1.0 - alpha) * q_curr[j] + alpha * q_next[j];
                }
            }

            setAllArmsQpos(model_, data_plan_, q_multi);
            mju_zero(data_plan_->qvel, model_->nv);
            mju_zero(data_plan_->qacc, model_->nv);
            mj_forward(model_, data_plan_);

            for (int ci = 0; ci < data_plan_->ncon; ++ci)
            {
                const mjContact &c = data_plan_->contact[ci];

                // If dist > 0.02, it's safe. If < 0.02, it's a conflict.
                if (c.dist > SAFETY_MARGIN)
                    continue;

                int a1 = geomToAgent(model_, c.geom1, num_agents_);
                int a2 = geomToAgent(model_, c.geom2, num_agents_);

                if (a1 >= 0 && a2 >= 0 && a1 != a2)
                {
                    out_conflict.agent1 = a1;
                    out_conflict.agent2 = a2;
                    out_conflict.t = static_cast<int>(t);
                    out_conflict.is_edge_conflict = false;

                    std::cout << "[CBS] Conflict found: Agent " << a1 << " vs Agent " << a2
                              << " at step " << t << " (alpha=" << alpha << ")" << std::endl;
                    return true;
                }
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
            return nullptr;

        child->paths[agent_to_constrain] = std::move(new_path);
        child->cost = computeTotalCost(child->paths);
        child->depth = node->depth + 1;

        return child;
    };

    auto child_i = make_child(i);
    if (child_i)
        open.push(child_i);

    auto child_j = make_child(j);
    if (child_j)
        open.push(child_j);
}

bool CBSPlanner::lowLevelPlan(int agent_id,
                              const JointConfig &q_start,
                              const JointConfig &q_goal,
                              const std::vector<Constraint> &all_constraints,
                              AgentPath &out_path)
{
    std::srand(static_cast<unsigned>(std::time(nullptr)) + agent_id); // Offset seed by agent

    std::vector<std::vector<double>> q_start_multi(num_agents_, std::vector<double>(dof_, 0.0));
    std::vector<std::vector<double>> q_goal_multi(num_agents_, std::vector<double>(dof_, 0.0));

    if ((int)start_configs_.size() == num_agents_)
    {
        for (int a = 0; a < num_agents_; ++a)
        {
            q_start_multi[a] = start_configs_[a];
            q_goal_multi[a] = start_configs_[a];
        }
    }

    q_start_multi[agent_id] = q_start;
    q_goal_multi[agent_id] = q_goal;

    Node start_node(q_start_multi, 0.0);
    Node goal_node(q_goal_multi, 1.0);

    // --- RETRY LOOP ---
    // Try RRT multiple times because it is random.
    const int MAX_RETRIES = 10;
    const int RRT_ITERS = 5000;
    const double STEP_SIZE = 0.5;

    for (int attempt = 0; attempt < MAX_RETRIES; ++attempt)
    {
        std::vector<Node *> rrt_path =
            rrtConnect(model_,
                       num_agents_,
                       dof_,
                       &start_node,
                       &goal_node,
                       RRT_ITERS,
                       STEP_SIZE,
                       agent_id);

        if (rrt_path.empty())
        {
            continue;
        }

        AgentPath candidate;
        candidate.reserve(rrt_path.size());
        bool extraction_ok = true;
        for (Node *n : rrt_path)
        {
            if ((int)n->q.size() <= agent_id)
            {
                extraction_ok = false;
                break;
            }
            candidate.push_back(n->q[agent_id]);
        }

        for (Node *n : rrt_path)
            delete n;

        if (!extraction_ok)
            continue;

        bool violation = false;
        for (size_t t = 0; t < candidate.size(); ++t)
        {
            const JointConfig &q = candidate[t];
            if (violatesVertexConstraints(agent_id, q, static_cast<int>(t), all_constraints))
            {
                violation = true;
                break;
            }
        }

        if (!violation)
        {
            out_path = std::move(candidate);
            return true;
        }
    }

    // If we tried MAX_RETRIES times and failed every time:
    // std::cerr << "[CBS] Failed to find valid low-level path for agent " << agent_id << " after retries.\n";
    return false;
}

bool CBSPlanner::debugLowLevelPlan(int agent_id, const JointConfig &q_start, const JointConfig &q_goal, const std::vector<Constraint> &constraints, AgentPath &out_path)
{
    start_configs_.assign(num_agents_, std::vector<double>(dof_, 0.0));
    start_configs_[agent_id] = q_start;
    return lowLevelPlan(agent_id, q_start, q_goal, constraints, out_path);
}

std::vector<Node *> CBSPlanner::plan(const std::vector<std::vector<double>> &start_poses, const std::vector<std::vector<double>> &goal_poses)
{
    if ((int)start_poses.size() != num_agents_ || (int)goal_poses.size() != num_agents_)
    {
        std::cerr << "[CBS] Error: start_poses / goal_poses size mismatch with num_agents\n";
        return {};
    }

    start_configs_.assign(start_poses.begin(), start_poses.end());
    goal_configs_.assign(goal_poses.begin(), goal_poses.end());

    auto root = make_shared<CTNode>();
    root->constraints.clear();
    root->paths.resize(num_agents_);

    for (int agent = 0; agent < num_agents_; ++agent)
    {
        AgentPath path_i;
        bool ok = lowLevelPlan(agent, start_poses[agent], goal_poses[agent], root->constraints, path_i);
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

    int iter = 0;
    while (!open.empty())
    {
        auto node = open.top();
        open.pop();
        iter++;

        Conflict conflict;
        bool has_conflict = findFirstConflict(node->paths, conflict);

        if (!has_conflict)
        {
            std::cout << "[CBS] Found conflict-free solution with cost " << node->cost << " after " << iter << " iterations.\n";
            return buildNodeTrajectory(node->paths);
        }

        if (iter > 100)
        {
            std::cout << "[CBS] Max iterations reached.\n";
            break;
        }

        expandNode(node, conflict, open);
    }

    std::cerr << "[CBS] No solution found.\n";
    return {};
}