#include "rrt_connect.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>

using std::vector;
struct TreeNode
{
    Node *cfg;
    TreeNode *parent;
};

static vector<double> flatten(const vector<vector<double>> &q)
{
    vector<double> flat;
    for (const auto &arm : q)
        flat.insert(flat.end(), arm.begin(), arm.end());
    return flat;
}
static double distance(const vector<vector<double>> &a, const vector<vector<double>> &b)
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

static bool isConfigValid(mjModel *model, mjData *data, const vector<vector<double>> &q, int agent_id)
{
    return isStateValid(model, data, q, false, agent_id);
}

static Node *sampleRandom(int num_actuators, int dof, int agent_id, const vector<vector<double>> &fixed_q)
{
    vector<vector<double>> q(num_actuators, vector<double>(dof, 0.0));
    const double qmin = -2.8;
    const double qmax = 2.8;
    for (int arm = 0; arm < num_actuators; ++arm)
    {
        if (agent_id != -1 && arm != agent_id)
        {
            q[arm] = fixed_q[arm];
            continue;
        }
        for (int j = 0; j < dof; ++j)
        {
            double u = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
            q[arm][j] = qmin + u * (qmax - qmin);
        }
    }
    return new Node(q, 0.0);
}

static Node *connect(const Node *from, const Node *to, double step_size)
{
    double dist = distance(from->q, to->q);
    if (dist <= step_size)
        return new Node(to->q, 0.0);
    double alpha = step_size / dist;
    int num_actuators = from->q.size();
    int dof = from->q[0].size();
    vector<vector<double>> q(num_actuators, vector<double>(dof, 0.0));
    for (int arm = 0; arm < num_actuators; ++arm)
    {
        for (int j = 0; j < dof; ++j)
            q[arm][j] = from->q[arm][j] + alpha * (to->q[arm][j] - from->q[arm][j]);
    }
    return new Node(q, 0.0);
}

static bool edgeCollisionFree(mjModel *model, mjData *data, const Node *a, const Node *b, double step_size, int agent_id)
{
    double dist = distance(a->q, b->q);
    double interp_step_size = 0.1;
    int steps = std::max(1, static_cast<int>(std::ceil(dist / interp_step_size)));
    for (int k = 0; k <= steps; ++k)
    {
        double alpha = static_cast<double>(k) / static_cast<double>(steps);
        Node interp(a->q, 0.0);
        int num_actuators = a->q.size();
        int dof = a->q[0].size();
        for (int arm = 0; arm < num_actuators; ++arm)
        {
            for (int j = 0; j < dof; ++j)
                interp.q[arm][j] = (1.0 - alpha) * a->q[arm][j] + alpha * b->q[arm][j];
        }
        if (!isConfigValid(model, data, interp.q, agent_id))
            return false;
    }
    return true;
}

enum ExtendStatus
{
    Trapped,
    Advanced,
    Reached
};

static TreeNode *nearest(const vector<TreeNode *> &tree, const Node *target)
{
    TreeNode *best = nullptr;
    double best_dist = 1e9;
    for (TreeNode *n : tree)
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

static ExtendStatus extend(TreeNode *&new_node, vector<TreeNode *> &tree, const Node *target, mjModel *model, mjData *data, double step_size, int agent_id)
{
    TreeNode *nearest_node = nearest(tree, target);
    Node *stepped = connect(nearest_node->cfg, target, step_size);

    if (!edgeCollisionFree(model, data, nearest_node->cfg, stepped, step_size, agent_id))
    {
        delete stepped;
        return Trapped;
    }
    //std::cout << "generated collision free edge" << std::endl;
    new_node = new TreeNode{stepped, nearest_node};
    tree.push_back(new_node);
    if (distance(stepped->q, target->q) < 1e-3)
        return Reached;
    return Advanced;
}

static vector<Node *> buildPath(TreeNode *a, TreeNode *b, const Node *exact_goal)
{
    vector<Node *> path_a;
    for (TreeNode *n = a; n; n = n->parent)
        path_a.push_back(new Node(n->cfg->q, 0.0));
    std::reverse(path_a.begin(), path_a.end());

    vector<Node *> path_b;
    for (TreeNode *n = b; n; n = n->parent)
        path_b.push_back(new Node(n->cfg->q, 0.0));

    if (!path_b.empty())
        path_b.pop_back();

    std::reverse(path_b.begin(), path_b.end());

    path_a.insert(path_a.end(), path_b.begin(), path_b.end());

    if (!path_a.empty())
    {
        // Overwrite the last configuration with the exact goal configuration
        path_a.back()->q = exact_goal->q;
    }

    for (size_t i = 0; i < path_a.size(); ++i)
        path_a[i]->t = static_cast<double>(i);

    return path_a;
}

vector<Node *> rrtConnect(mjModel *model, int num_actuators, int dof, const Node *start, const Node *goal, int max_iters, double step_size, int agent_id)
{
    mjData *data = mj_makeData(model);
    vector<TreeNode *> Ta, Tb;
    Ta.push_back(new TreeNode{new Node(start->q, 0.0), nullptr});
    Tb.push_back(new TreeNode{new Node(goal->q, 0.0), nullptr});

    for (int iter = 0; iter < max_iters; ++iter)
    {
        Node *q_rand = sampleRandom(num_actuators, dof, agent_id, start->q);
        TreeNode *a_new = nullptr;

        if (extend(a_new, Ta, q_rand, model, data, step_size, agent_id) != Trapped)
        {
            TreeNode *b_new = nullptr;
            ExtendStatus status = Advanced;
            while (status == Advanced)
                status = extend(b_new, Tb, a_new->cfg, model, data, step_size, agent_id);
            if (status == Reached)
            {
                vector<Node *> path = buildPath(a_new, b_new, start);
                delete q_rand;
                mj_deleteData(data);
                return path;
            }
        }
        delete q_rand;
        std::swap(Ta, Tb);
    }
    std::printf("RRT-Connect: no path after %d iterations\n", max_iters);
    mj_deleteData(data);
    return {};
}