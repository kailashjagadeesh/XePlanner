#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <queue>
#include <memory>
#include "utils.h"
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cmath>

#ifndef PLANNER_SHARED_TYPES
#define PLANNER_SHARED_TYPES
enum class ConstraintType
{
    VERTEX,
    EDGE
};

struct Constraint
{
    int agent_id;
    ConstraintType type;
    int t;

    // For vertex constraint
    std::vector<double> q;

    // For edge constraint
    std::vector<double> q_from;
    std::vector<double> q_to;
};

struct Conflict
{
    int agent1;
    int agent2;
    int t;
    bool is_edge_conflict;
};

using JointConfig = std::vector<double>;
using AgentPath = std::vector<JointConfig>;
using MultiAgentPaths = std::vector<AgentPath>;

struct CTNode
{
    std::vector<Constraint> constraints; // N.C
    MultiAgentPaths paths;               // N.pi
    double cost = 0.0;
    int num_conflicts = 0; 

    int depth = 0;
};

struct CTNodeCompare
{
    bool operator()(const std::shared_ptr<CTNode> &a,
                    const std::shared_ptr<CTNode> &b) const
    {
        return a->cost > b->cost;
    }
};

struct SingleArmNode
{
    JointConfig q;
    double t;
    std::vector<SingleArmNode *> parents;
    SingleArmNode(const std::vector<double> &q_, double t_) : q(q_), t(t_) {}
};

struct SingleArmNodeHasher
{
    std::size_t operator()(const SingleArmNode *n) const noexcept
    {
        std::size_t h = 0;
        std::hash<double> dh;

        for (double q : n->q)
        {
            // Combine hashes
            h ^= dh(q) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }

        // Include time
        h ^= dh(n->t) + 0x9e3779b9 + (h << 6) + (h >> 2);

        return h;
    }
};

struct SingleArmNodeEqual
{
    bool operator()(const SingleArmNode *a, const SingleArmNode *b) const noexcept
    {
        if (a->q.size() != b->q.size())
            return false;
        if (a->t != b->t)
            return false;

        for (size_t i = 0; i < a->q.size(); ++i)
        {
            if (std::abs(a->q[i] - b->q[i]) > 0.05) // this should match dq in get successors
                return false;
        }

        return true;
    }
};
#endif // PLANNER_SHARED_TYPES

// Experience entry for caching low-level plans
struct ExperienceEntry
{
    int agent_id;
    JointConfig q_start;
    JointConfig q_goal;
    std::vector<Constraint> constraints;
    AgentPath path;
    double cost;
    
    // Hash key for quick lookup
    std::size_t hash_key;
};

// Helper function to compute constraint hash (order-independent)
static std::size_t hashConstraint(const Constraint &c)
{
    std::size_t h = 0;
    std::hash<double> dh;
    std::hash<int> ih;
    
    h ^= ih(c.agent_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= ih(static_cast<int>(c.type)) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= ih(c.t) + 0x9e3779b9 + (h << 6) + (h >> 2);
    
    // Hash constraint configurations
    if (c.type == ConstraintType::VERTEX)
    {
        for (double q : c.q)
        {
            h ^= dh(q) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
    }
    else // EDGE constraint
    {
        for (double q : c.q_from)
        {
            h ^= dh(q) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        for (double q : c.q_to)
        {
            h ^= dh(q) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
    }
    
    return h;
}

struct ExperienceHasher
{
    std::size_t operator()(const ExperienceEntry &e) const noexcept
    {
        std::size_t h = 0;
        std::hash<double> dh;
        std::hash<int> ih;
        
        // Hash agent_id
        h ^= ih(e.agent_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        
        // Hash start config
        for (double q : e.q_start)
        {
            h ^= dh(q) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        
        // Hash goal config
        for (double q : e.q_goal)
        {
            h ^= dh(q) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        
        // Hash constraints (order-independent: sum of individual constraint hashes)
        h ^= ih(e.constraints.size()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        std::size_t constraint_hash_sum = 0;
        for (const auto &c : e.constraints)
        {
            constraint_hash_sum += hashConstraint(c);
        }
        h ^= constraint_hash_sum + 0x9e3779b9 + (h << 6) + (h >> 2);
        
        return h;
    }
};

// Helper function to compare two constraints (order-independent comparison)
static bool constraintsEqual(const Constraint &ca, const Constraint &cb)
{
    if (ca.agent_id != cb.agent_id || ca.type != cb.type || ca.t != cb.t)
        return false;
    
    const double EPSILON = 0.1;
    
    // Compare constraint configurations
    if (ca.type == ConstraintType::VERTEX)
    {
        if (ca.q.size() != cb.q.size())
            return false;
        for (size_t j = 0; j < ca.q.size(); ++j)
        {
            if (std::fabs(ca.q[j] - cb.q[j]) > EPSILON)
                return false;
        }
    }
    else // EDGE constraint
    {
        if (ca.q_from.size() != cb.q_from.size() || ca.q_to.size() != cb.q_to.size())
            return false;
        for (size_t j = 0; j < ca.q_from.size(); ++j)
        {
            if (std::fabs(ca.q_from[j] - cb.q_from[j]) > EPSILON)
                return false;
        }
        for (size_t j = 0; j < ca.q_to.size(); ++j)
        {
            if (std::fabs(ca.q_to[j] - cb.q_to[j]) > EPSILON)
                return false;
        }
    }
    
    return true;
}

struct ExperienceEqual
{
    bool operator()(const ExperienceEntry &a, const ExperienceEntry &b) const noexcept
    {
        if (a.agent_id != b.agent_id)
            return false;
        if (a.q_start.size() != b.q_start.size() || a.q_goal.size() != b.q_goal.size())
            return false;
        if (a.constraints.size() != b.constraints.size())
            return false;
        
        const double EPSILON = 0.1;
        for (size_t i = 0; i < a.q_start.size(); ++i)
        {
            if (std::fabs(a.q_start[i] - b.q_start[i]) > EPSILON)
                return false;
        }
        for (size_t i = 0; i < a.q_goal.size(); ++i)
        {
            if (std::fabs(a.q_goal[i] - b.q_goal[i]) > EPSILON)
                return false;
        }
        
        // Compare constraints (order-independent: check if each constraint in a has a match in b)
        for (const auto &ca : a.constraints)
        {
            bool found_match = false;
            for (const auto &cb : b.constraints)
            {
                if (constraintsEqual(ca, cb))
                {
                    found_match = true;
                    break;
                }
            }
            if (!found_match)
                return false;
        }
        
        return true;
    }
};

class XeCBSPlanner
{
public:
    XeCBSPlanner(mjModel *model,
                 double dq_max,
                 double dt,
                 int num_agents,
                 int dof,
                 std::vector<int> &body_to_arm,
                 std::vector<std::vector<int>> &joint_id,
                 double suboptimal_factor=1.5);

    ~XeCBSPlanner();

    // Return std::vector<Node*>  A multi-arm trajectory over time (Node.q = [agent][joint]). Returns empty vector if no solution is found.
    std::vector<Node *> plan(const std::vector<std::vector<double>> &start_poses,
                             const std::vector<std::vector<double>> &goal_poses);

    bool debugLowLevelPlan(int agent_id,
                           const JointConfig &q_start,
                           const JointConfig &q_goal,
                           const std::vector<Constraint> &constraints,
                           AgentPath &out_path);

private:
    bool findFirstConflict(const MultiAgentPaths &paths, Conflict &out_conflict);
    bool findAllConflicts(const MultiAgentPaths &paths, std::vector<Conflict> &out_conflicts);

    void expandNode(const std::shared_ptr<CTNode> &node,
                    const Conflict &conflict,
                    std::priority_queue<std::shared_ptr<CTNode>,
                                        std::vector<std::shared_ptr<CTNode>>,
                                        CTNodeCompare> &open);

    bool lowLevelPlan(int agent_id,
                      const JointConfig &q_start,
                      const JointConfig &q_goal,
                      const std::vector<Constraint> &constraints,
                      AgentPath &out_path);

    std::vector<SingleArmNode*> get_successors(SingleArmNode* current);

    // Utility: extract constraints relevant to one agent.
    std::vector<Constraint> getConstraintsForAgent(int agent_id, const std::vector<Constraint> &all_constraints) const;

    // Compute sum of path costs for a CT node.
    double computeTotalCost(const MultiAgentPaths &paths) const;

    // Convert final MultiAgentPaths to vector<Node*> compatible with your simulator.
    std::vector<Node *> buildNodeTrajectory(const MultiAgentPaths &paths) const;

    // XECBS-specific: Experience management
    bool queryExperience(int agent_id,
                         const JointConfig &q_start,
                         const JointConfig &q_goal,
                         const std::vector<Constraint> &constraints,
                         AgentPath &out_path);
    
    void storeExperience(int agent_id,
                         const JointConfig &q_start,
                         const JointConfig &q_goal,
                         const std::vector<Constraint> &constraints,
                         const AgentPath &path);
    
    std::size_t computeExperienceHash(int agent_id,
                                      const JointConfig &q_start,
                                      const JointConfig &q_goal) const;

private:
    mjModel *model_;
    mjData *data_plan_;
    double dq_max_;
    double dt_;
    int num_agents_;
    int dof_;
    std::vector<int> body_to_arm_;
    std::vector<std::vector<int>> joint_id_;
    double suboptimal_factor_; // weight for the suboptimal cost
    double best_cost_; // best cost found so far
    std::vector<JointConfig> start_configs_;
    std::vector<JointConfig> goal_configs_;
    
    // XECBS-specific: Experience database
    std::unordered_multimap<std::size_t, ExperienceEntry> experience_db_;
    int experience_hits_;  // Statistics: number of experience cache hits
    int experience_misses_; // Statistics: number of experience cache misses
};

