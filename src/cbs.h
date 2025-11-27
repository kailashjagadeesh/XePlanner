#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <queue>
#include <memory>
#include "utils.h"

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

class CBSPlanner
{
public:
    CBSPlanner(mjModel *model,
               double dq_max,
               double dt,
               int num_agents,
               int dof);

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

    std::vector<JointConfig> start_configs_;
    std::vector<JointConfig> goal_configs_;
};
