#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <queue>
#include <memory>
#include "utils.h"
#include <unordered_set>
#include <unordered_map>

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
            if (a->q[i] != b->q[i])
                return false;
        }

        return true;
    }
};
#endif // PLANNER_SHARED_TYPES

class CBSPlanner
{
public:
    CBSPlanner(mjModel *model,
               double dq_max,
               double dt,
               int num_agents,
               int dof,
               std::vector<int> &body_to_arm,
               std::vector<std::vector<int>> &joint_id);

    ~CBSPlanner();

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

private:
    mjModel *model_;
    mjData *data_plan_;
    double dq_max_;
    double dt_;
    int num_agents_;
    int dof_;
    std::vector<int> body_to_arm_;
    std::vector<std::vector<int>> joint_id_;

    std::vector<JointConfig> start_configs_;
    std::vector<JointConfig> goal_configs_;
};
