#include "rrt_connect.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>

using std::vector;

struct TreeNode
{
    Node* cfg;
    TreeNode* parent;
};

// flattens the q vector for all arms so that distance can be easily computed.
static vector<double> flatten(const vector<vector<double>>& q)
{
    vector<double> flat;
    for (const auto& arm : q)
    {
        flat.insert(flat.end(), arm.begin(), arm.end());
    }
    return flat;
}

// Eucledian distance between joint angles
static double distance(const vector<vector<double>>& a, const vector<vector<double>>& b)
{
    auto fa = flatten(a);
    auto fb = flatten(b);
    double sum = 0.0;
    for (size_t i = 0; i < fa.size(); ++i)
    {
        double d = fa[i] - fb[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

// Set model joint positions to q and check collision.
static bool isConfigValid(mjModel* model, mjData* data, const vector<vector<double>>& q)
{
    int num_actuators = q.size();
    int dof = q[0].size();

    for (int arm = 0; arm < num_actuators; ++arm)
    {
        for (int j = 0; j < dof; ++j)
        {
            char name[64];
            std::snprintf(name, sizeof(name), "panda%d_joint%d", arm + 1, j + 1);
            int id = mj_name2id(model, mjOBJ_JOINT, name);
            if (id < 0)
            {
                return false;
            }
            int qpos_adr = model->jnt_qposadr[id];
            data->qpos[qpos_adr] = q[arm][j];
        }
    }

    mj_forward(model, data);
    return !hasCollision(model, data, false);
}

// Sample a random configuration within simple symmetric joint limits.
static Node* sampleRandom(int num_actuators, int dof)
{
    vector<vector<double>> q(num_actuators, vector<double>(dof, 0.0));
    const double qmin = -2.8;
    const double qmax = 2.8;

    for (int arm = 0; arm < num_actuators; ++arm)
    {
        for (int j = 0; j < dof; ++j)
        {
            double u = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
            q[arm][j] = qmin + u * (qmax - qmin);
        }
    }
    return new Node(q, 0.0);
}

// Move from "from" toward "to" by at most step_size in configuration distance.
static Node* connect(const Node* from, const Node* to, double step_size)
{
    double dist = distance(from->q, to->q);
    if (dist <= step_size)
    {
        return new Node(to->q, 0.0);
    }

    double alpha = step_size / dist;
    int num_actuators = from->q.size();
    int dof = from->q[0].size();
    vector<vector<double>> q(num_actuators, vector<double>(dof, 0.0));

    for (int arm = 0; arm < num_actuators; ++arm)
    {
        for (int j = 0; j < dof; ++j)
        {
            q[arm][j] = from->q[arm][j] + alpha * (to->q[arm][j] - from->q[arm][j]);
        }
    }
    return new Node(q, 0.0);
}

// Check that the straight line from a to b is collision-free with small steps.
static bool edgeCollisionFree(mjModel* model, mjData* data, const Node* a, const Node* b, double step_size)
{
    double dist = distance(a->q, b->q);
    int steps = std::max(1, static_cast<int>(std::ceil(dist / step_size)));

    for (int k = 0; k <= steps; ++k)
    {
        double alpha = static_cast<double>(k) / static_cast<double>(steps);
        Node interp(a->q, 0.0);
        int num_actuators = a->q.size();
        int dof = a->q[0].size();
        for (int arm = 0; arm < num_actuators; ++arm)
        {
            for (int j = 0; j < dof; ++j)
            {
                interp.q[arm][j] = (1.0 - alpha) * a->q[arm][j] + alpha * b->q[arm][j];
            }
        }

        if (!isConfigValid(model, data, interp.q))
        {
            return false;
        }
    }
    return true;
}

enum ExtendStatus
{
    Trapped,
    Advanced,
    Reached
};

static TreeNode* nearest(const vector<TreeNode*>& tree, const Node* target)
{
    TreeNode* best = nullptr;
    double best_dist = 1e9;
    for (TreeNode* n : tree)
    {
        double d = distance(n->cfg->q, target->q);
        if (d < best_dist)
        {
            best_dist = d;
            best = n;
        }
    }
    return best;
}

static ExtendStatus extend(TreeNode*& new_node,
                           vector<TreeNode*>& tree,
                           const Node* target,
                           mjModel* model,
                           mjData* data,
                           double step_size)
{
    TreeNode* nearest_node = nearest(tree, target);
    Node* stepped = connect(nearest_node->cfg, target, step_size);

    if (!edgeCollisionFree(model, data, nearest_node->cfg, stepped, step_size))
    {
        delete stepped;
        return Trapped;
    }

    new_node = new TreeNode{stepped, nearest_node};
    tree.push_back(new_node);

    double remaining = distance(stepped->q, target->q);
    if (remaining < 1e-3)
    {
        return Reached;
    }
    return Advanced;
}

// Reconstruct path by following parents to the root.
static vector<Node*> buildPath(TreeNode* a, TreeNode* b)
{
    vector<Node*> path_a;
    for (TreeNode* n = a; n; n = n->parent)
    {
        path_a.push_back(new Node(n->cfg->q, 0.0));
    }
    std::reverse(path_a.begin(), path_a.end());

    vector<Node*> path_b;
    for (TreeNode* n = b; n; n = n->parent)
    {
        path_b.push_back(new Node(n->cfg->q, 0.0));
    }
    // Drop the meeting point from the second half to avoid duplication.
    if (!path_b.empty())
    {
        path_b.pop_back();
    }
    // path_b is goal back to meet; reverse to go forward.
    std::reverse(path_b.begin(), path_b.end());

    path_a.insert(path_a.end(), path_b.begin(), path_b.end());
    // Assign simple timestamps: 0,1,2,...
    for (size_t i = 0; i < path_a.size(); ++i)
    {
        path_a[i]->t = static_cast<double>(i);
    }
    return path_a;
}

vector<Node*> rrtConnect(mjModel* model,
                         int num_actuators,
                         int dof,
                         const Node* start,
                         const Node* goal,
                         int max_iters,
                         double step_size)
{
    mjData* data = mj_makeData(model);

    // Trees rooted at start and goal.
    vector<TreeNode*> Ta;
    vector<TreeNode*> Tb;
    Ta.push_back(new TreeNode{new Node(start->q, 0.0), nullptr});
    Tb.push_back(new TreeNode{new Node(goal->q, 0.0), nullptr});

    for (int iter = 0; iter < max_iters; ++iter)
    {
        Node* q_rand = sampleRandom(num_actuators, dof);

        TreeNode* a_new = nullptr;
        ExtendStatus s = extend(a_new, Ta, q_rand, model, data, step_size);
        if (s != Trapped)
        {
            TreeNode* b_new = nullptr;
            ExtendStatus status_connect = Advanced;
            while (status_connect == Advanced)
            {
                status_connect = extend(b_new, Tb, a_new->cfg, model, data, step_size);
            }

            if (status_connect == Reached)
            {
                std::printf("RRT-Connect: reached at iter %d\n", iter);
                vector<Node*> path = buildPath(a_new, b_new);
                delete q_rand;
                mj_deleteData(data);
                return path;
            }
        }

        delete q_rand;
        std::swap(Ta, Tb); // alternate which tree grows from random samples

        if ((iter + 1) % 100 == 0)
        {
            std::printf("RRT-Connect progress: iter %d (Ta size %zu, Tb size %zu)\n",
                        iter + 1, Ta.size(), Tb.size());
        }
    }

    std::printf("RRT-Connect: no path after %d iterations, using straight-line fallback\n", max_iters);
    mj_deleteData(data);
    // Fallback: straight line between start and goal.
    vector<Node*> fallback;
    fallback.push_back(new Node(start->q, 0.0));
    fallback.push_back(new Node(goal->q, 1.0));
    return fallback;
}
