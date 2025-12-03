#include "xecbs.h"
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

// FIXED: Added epsilon for float comparison
static bool violatesVertexConstraints(int agent_id,
                                      const JointConfig &q,
                                      int t,
                                      const std::vector<Constraint> &constraints)
{
    const double EPSILON = 0.1; // Tolerance in radians. RRT is noisy.

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

XeCBSPlanner::XeCBSPlanner(mjModel *model, double dq_max, double dt, int num_agents, int dof, std::vector<int> &body_to_arm, std::vector<std::vector<int>> &joint_id, double suboptimal_factor)
    : model_(model), dq_max_(dq_max), dt_(dt), num_agents_(num_agents), dof_(dof), body_to_arm_(body_to_arm), joint_id_(joint_id),
      suboptimal_factor_(suboptimal_factor), best_cost_(std::numeric_limits<double>::max()),
      experience_hits_(0), experience_misses_(0)
{
    if (!model_)
        throw std::runtime_error("XeCBSPlanner: model pointer is null");
    data_plan_ = mj_makeData(model_);
    if (!data_plan_)
        throw std::runtime_error("XeCBSPlanner: failed to allocate mjData for planning");
}

XeCBSPlanner::~XeCBSPlanner()
{
    if (data_plan_)
    {
        mj_deleteData(data_plan_);
        data_plan_ = nullptr;
    }
}

double XeCBSPlanner::computeTotalCost(const MultiAgentPaths &paths) const
{
    double total = 0.0;
    for (const auto &path : paths)
        total += static_cast<double>(path.size());
    return total;
}

std::vector<Node *> XeCBSPlanner::buildNodeTrajectory(const MultiAgentPaths &paths) const
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
    for (int t = 0; t < T_max - 1; t++)
    {
        std::vector<std::vector<double>> q_multi;
        for (int arm = 0; arm < paths.size(); arm++)
        {
            int step = std::min(t, static_cast<int>(paths[arm].size() - 1));
            q_multi.push_back(paths[arm][step]);
        }
        Node *node = new Node(q_multi, current_time);
        current_time += dt_;
        traj.push_back(node);
    }

    return traj;
}

std::vector<Constraint> XeCBSPlanner::getConstraintsForAgent(int agent_id, const std::vector<Constraint> &all_constraints) const
{
    std::vector<Constraint> result;
    for (const auto &c : all_constraints)
    {
        if (c.agent_id == agent_id)
            result.push_back(c);
    }
    return result;
}

bool XeCBSPlanner::findFirstConflict(const MultiAgentPaths &paths, Conflict &out_conflict)
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
        std::vector<std::vector<double>> waypoint;
        for (int arm = 0; arm < paths.size(); arm++)
        {
            int step = std::min(t, paths[arm].size() - 1);
            waypoint.push_back(paths[arm][step]);
        }
        std::vector<std::pair<int, int>> collisions = isMultiArmCollision(waypoint, model_, data_plan_, joint_id_, body_to_arm_, 0.2);
        if (collisions.size() > 0)
        {
            auto [arm1, arm2] = collisions[0];
            out_conflict.agent1 = arm1;
            out_conflict.agent2 = arm2;
            out_conflict.t = static_cast<int>(t);
            out_conflict.is_edge_conflict = false;
            return true;
        }
    }
    return false;
}

bool XeCBSPlanner::findAllConflicts(const MultiAgentPaths &paths, std::vector<Conflict> &out_conflicts)
{
    out_conflicts.clear();
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
        std::vector<std::vector<double>> waypoint;
        for (int arm = 0; arm < paths.size(); arm++)
        {
            int step = std::min(t, paths[arm].size() - 1);
            waypoint.push_back(paths[arm][step]);
        }
        std::vector<std::pair<int, int>> collisions = isMultiArmCollision(waypoint, model_, data_plan_, joint_id_, body_to_arm_, 0.2);
        for (const auto &[arm1, arm2] : collisions)
        {
            Conflict conflict;
            conflict.agent1 = arm1;
            conflict.agent2 = arm2;
            conflict.t = static_cast<int>(t);
            conflict.is_edge_conflict = false;
            out_conflicts.push_back(conflict);
        }
    }
    return !out_conflicts.empty();
}

void XeCBSPlanner::expandNode(const std::shared_ptr<CTNode> &node,
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
        // Update number of conflicts
        std::vector<Conflict> all_conflicts;
        findAllConflicts(child->paths, all_conflicts);
        child->num_conflicts = all_conflicts.size();

        return child;
    };

    auto child_i = make_child(i);
    if (child_i)
        open.push(child_i);

    auto child_j = make_child(j);
    if (child_j)
        open.push(child_j);
}

std::vector<SingleArmNode *> XeCBSPlanner::get_successors(SingleArmNode *current)
{
    // start with just doing single arm motions
    std::vector<SingleArmNode *> successors;
    successors.push_back(new SingleArmNode(current->q, current->t + dt_));
    double dq = 0.05; // 0.2 rads
    for (int j = 0; j < current->q.size(); j++)
    {
        SingleArmNode *successor = new SingleArmNode(current->q, current->t + dt_);
        successor->q[j] += dq;
        successors.push_back(successor);

        SingleArmNode *reverse_succesor = new SingleArmNode(current->q, current->t + dt_);
        reverse_succesor->q[j] -= dq;
        successors.push_back(reverse_succesor);
    }
    return successors;
}

// XECBS-specific: Compute hash for experience lookup
std::size_t XeCBSPlanner::computeExperienceHash(int agent_id,
                                                 const JointConfig &q_start,
                                                 const JointConfig &q_goal,
                                                 const std::vector<Constraint> &constraints) const
{
    ExperienceHasher hasher;
    ExperienceEntry temp_entry;
    temp_entry.agent_id = agent_id;
    temp_entry.q_start = q_start;
    temp_entry.q_goal = q_goal;
    temp_entry.constraints = constraints;
    return hasher(temp_entry);
}

// XECBS-specific: Query experience cache
bool XeCBSPlanner::queryExperience(int agent_id,
                                    const JointConfig &q_start,
                                    const JointConfig &q_goal,
                                    const std::vector<Constraint> &constraints,
                                    AgentPath &out_path)
{
    // Create a temporary entry for lookup
    ExperienceEntry query_entry;
    query_entry.agent_id = agent_id;
    query_entry.q_start = q_start;
    query_entry.q_goal = q_goal;
    query_entry.constraints = constraints;
    
    // Try to find matching experience
    auto it = experience_db_.find(query_entry);
    if (it != experience_db_.end())
    {
        // Cache hit! Return the cached path
        out_path = it->path;
        experience_hits_++;
        return true;
    }
    
    // Cache miss
    experience_misses_++;
    return false;
}

// XECBS-specific: Store experience in cache
void XeCBSPlanner::storeExperience(int agent_id,
                                    const JointConfig &q_start,
                                    const JointConfig &q_goal,
                                    const std::vector<Constraint> &constraints,
                                    const AgentPath &path)
{
    ExperienceEntry entry;
    entry.agent_id = agent_id;
    entry.q_start = q_start;
    entry.q_goal = q_goal;
    entry.constraints = constraints;
    entry.path = path;
    entry.cost = static_cast<double>(path.size());
    entry.hash_key = computeExperienceHash(agent_id, q_start, q_goal, constraints);
    
    // Insert into experience database
    // Note: unordered_set will automatically handle duplicates
    experience_db_.insert(entry);
}

bool XeCBSPlanner::lowLevelPlan(int agent_id,
                              const JointConfig &q_start,
                              const JointConfig &q_goal,
                              const std::vector<Constraint> &all_constraints,
                              AgentPath &out_path)
{
    // XECBS: First check experience cache
    if (queryExperience(agent_id, q_start, q_goal, all_constraints, out_path))
    {
        // Cache hit - path retrieved from experience
        return true;
    }
    
    // Cache miss - need to compute new path
    std::priority_queue<std::pair<double, SingleArmNode *>, std::vector<std::pair<double, SingleArmNode *>>, std::greater<std::pair<double, SingleArmNode *>>> open_list;
    std::unordered_set<SingleArmNode *, SingleArmNodeHasher, SingleArmNodeEqual> closed_set;
    std::unordered_map<SingleArmNode *, double, SingleArmNodeHasher, SingleArmNodeEqual> g_vals;

    SingleArmNode *start = new SingleArmNode(q_start, 0);
    open_list.push({0.0, start});
    g_vals[start] = 0.0;

    int max_expansions = 100000;
    AgentPath plan;
    bool success = false;
    std::vector<SingleArmNode *> nodes;
    nodes.push_back(start);
    while (!open_list.empty())
    {
        if (max_expansions == 0)
            break;
        auto [f_val, curr_node] = open_list.top();
        open_list.pop();

        if (closed_set.count(curr_node))
            continue;
        closed_set.insert(curr_node);
        max_expansions--;

        int at_goal = true;
        for (int j = 0; j < q_goal.size(); j++)
        {
            if (std::abs(curr_node->q[j] - q_goal[j]) > 0.05)
            {
                at_goal = false;
            }
        }
        if (at_goal)
        {
            success = true;
            SingleArmNode *current = curr_node;
            plan.push_back(current->q);
            while (current->parents.size() != 0)
            {
                SingleArmNode *best_parent = nullptr;
                for (SingleArmNode *parent : current->parents)
                {
                    if (best_parent == nullptr || g_vals[parent] < g_vals[best_parent])
                    {
                        best_parent = parent;
                    }
                }
                plan.push_back(best_parent->q);
                current = best_parent;
            }
            break;
        }

        std::vector<SingleArmNode *> successors = get_successors(curr_node);
        for (SingleArmNode *successor : successors)
        {
            if (isSingleArmCollision(successor->q, model_, data_plan_, agent_id, joint_id_, body_to_arm_) || violatesVertexConstraints(agent_id, successor->q, successor->t, all_constraints))
            {
                continue;
            }
            if (!g_vals.count(successor) || g_vals[successor] > g_vals[curr_node] + 1) // just using unit cost
            {
                if (!g_vals.count(successor))
                {
                    nodes.push_back(successor);
                }
                g_vals[successor] = g_vals[curr_node] + 1;
                successor->parents.push_back(curr_node);
                double h = heuristicL1(curr_node->q, q_goal);
                open_list.push({g_vals[successor] + 100 * h, successor});
            }
        }
    }

    for (int i = 0; i < nodes.size(); i++)
    {
        delete nodes[i];
    }

    if (success)
    {
        std::reverse(plan.begin(), plan.end());
        out_path = std::move(plan);
        
        // XECBS: Store the newly computed path in experience cache
        storeExperience(agent_id, q_start, q_goal, all_constraints, out_path);
        
        return true;
    }

    return false;
}

std::vector<Node *> XeCBSPlanner::plan(const std::vector<std::vector<double>> &start_poses, const std::vector<std::vector<double>> &goal_poses)
{
    if ((int)start_poses.size() != num_agents_ || (int)goal_poses.size() != num_agents_)
    {
        std::cerr << "[XeCBS] Error: start_poses / goal_poses size mismatch with num_agents\n";
        return {};
    }

    // Reset experience statistics for this planning session
    experience_hits_ = 0;
    experience_misses_ = 0;

    start_configs_.assign(start_poses.begin(), start_poses.end());
    goal_configs_.assign(goal_poses.begin(), goal_poses.end());

    // Initialize root node (same as ECBS)
    auto root = make_shared<CTNode>();
    root->constraints.clear();
    root->paths.resize(num_agents_);

    for (int agent = 0; agent < num_agents_; ++agent)
    {
        AgentPath path_i;
        bool ok = lowLevelPlan(agent, start_poses[agent], goal_poses[agent], root->constraints, path_i);
        if (!ok)
        {
            std::cerr << "[XeCBS] Root low-level plan failed for agent " << agent << "\n";
            return {};
        }
        root->paths[agent] = std::move(path_i);
    }

    root->cost = computeTotalCost(root->paths);
    root->depth = 0;

    std::priority_queue<shared_ptr<CTNode>, std::vector<shared_ptr<CTNode>>, CTNodeCompare> open;
    open.push(root);
    
    best_cost_ = root->cost;  // Initialize best cost

    int iter = 0;
    while (!open.empty())
    {
        // Update best_cost_ to minimum cost in OPEN
        if (!open.empty())
        {
            best_cost_ = std::min(best_cost_, open.top()->cost);
        }

        // ECBS FOCAL SEARCH: Build focal set (nodes with cost ≤ w * best_cost_)
        double focal_threshold = suboptimal_factor_ * best_cost_;
        std::vector<shared_ptr<CTNode>> focal;
        
        // Extract all nodes in focal set from OPEN, track minimum cost
        std::vector<shared_ptr<CTNode>> temp_storage;
        double min_cost_in_open = std::numeric_limits<double>::max();

        while (!open.empty())
        {
            auto node = open.top();
            open.pop();

            min_cost_in_open = std::min(min_cost_in_open, node->cost);
            
            if (node->cost <= focal_threshold)
            {
                focal.push_back(node);
            }
            else
            {
                temp_storage.push_back(node);
            }
        }
        
        // Update best_cost_ with the actual minimum from OPEN
        if (min_cost_in_open < std::numeric_limits<double>::max())
        {
            best_cost_ = std::min(best_cost_, min_cost_in_open);
        }

        // Re-insert nodes not in focal set
        for (auto node : temp_storage)
        {
            open.push(node);
        }

        // If focal set is empty, expand from OPEN
        if (focal.empty())
        {
            if (open.empty())
                break;
            auto node = open.top();
            open.pop();
            iter++;

            Conflict conflict;
            bool has_conflict = findFirstConflict(node->paths, conflict);

            if (!has_conflict)
            {
                std::cout << "[XeCBS] Found conflict-free solution with cost " << node->cost 
                         << " (suboptimality: " << (best_cost_ > 0 ? (node->cost / best_cost_) : 1.0) << ") after " << iter << " iterations.\n";
                std::cout << "[XeCBS] Experience cache: " << experience_hits_ << " hits, " << experience_misses_ << " misses\n";
                return buildNodeTrajectory(node->paths);
            }

            if (iter > 100)
            {
                std::cout << "[XeCBS] Max iterations reached.\n";
                break;
            }

            expandNode(node, conflict, open);
            continue;
        }

        // ECBS: Choose node from focal set with minimum number of conflicts
        // (Prioritize nodes that are likely to resolve conflicts faster)
        shared_ptr<CTNode> best_focal_node = nullptr;
        int min_conflicts = std::numeric_limits<int>::max();
        
        for (auto node : focal)
        {
            if (node->num_conflicts < min_conflicts)
            {
                min_conflicts = node->num_conflicts;
                best_focal_node = node;
            }
        }

        if (!best_focal_node)
        {
            best_focal_node = focal[0];  // Fallback
        }

        // Remove chosen node from focal set
        focal.erase(std::remove(focal.begin(), focal.end(), best_focal_node), focal.end());
        
        // Re-insert remaining focal nodes back to OPEN
        for (auto node : focal)
        {
            open.push(node);
        }

        iter++;

        Conflict conflict;
        bool has_conflict = findFirstConflict(best_focal_node->paths, conflict);

        if (!has_conflict)
        {
            std::cout << "[XeCBS] Found conflict-free solution with cost " << best_focal_node->cost 
                     << " (suboptimality: " << (best_cost_ > 0 ? (best_focal_node->cost / best_cost_) : 1.0) << ") after " << iter << " iterations.\n";
            std::cout << "[XeCBS] Experience cache: " << experience_hits_ << " hits, " << experience_misses_ << " misses\n";
            return buildNodeTrajectory(best_focal_node->paths);
        }

        if (iter > 100)
        {
            std::cout << "[XeCBS] Max iterations reached.\n";
            break;
        }

        expandNode(best_focal_node, conflict, open);
    }

    std::cerr << "[XeCBS] No solution found.\n";
    std::cout << "[XeCBS] Experience cache: " << experience_hits_ << " hits, " << experience_misses_ << " misses\n";
    return {};
}

bool XeCBSPlanner::debugLowLevelPlan(int agent_id,
                                      const JointConfig &q_start,
                                      const JointConfig &q_goal,
                                      const std::vector<Constraint> &constraints,
                                      AgentPath &out_path)
{
    return lowLevelPlan(agent_id, q_start, q_goal, constraints, out_path);
}

