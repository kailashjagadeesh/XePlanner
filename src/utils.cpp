#include "utils.h"
#include <iostream>
using namespace std;

void setArmActuatorTargets(mjModel *model, mjData *data, const vector<double> &target_pose)
{
    // Sets the position targets of all arm actuators
    const char *arm_actuators[][7] = {
        {"panda1_actuator1", "panda1_actuator2", "panda1_actuator3", "panda1_actuator4", "panda1_actuator5", "panda1_actuator6", "panda1_actuator7"},
        {"panda2_actuator1", "panda2_actuator2", "panda2_actuator3", "panda2_actuator4", "panda2_actuator5", "panda2_actuator6", "panda2_actuator7"},
        {"panda3_actuator1", "panda3_actuator2", "panda3_actuator3", "panda3_actuator4", "panda3_actuator5", "panda3_actuator6", "panda3_actuator7"},
        {"panda4_actuator1", "panda4_actuator2", "panda4_actuator3", "panda4_actuator4", "panda4_actuator5", "panda4_actuator6", "panda4_actuator7"}};

    for (const auto &act_list : arm_actuators)
    {
        for (int i = 0; i < 7; ++i)
        {
            int a_id = mj_name2id(model, mjOBJ_ACTUATOR, act_list[i]);
            if (a_id >= 0)
            {
                data->ctrl[a_id] = target_pose[i];
            }
        }
    }
}

void setGrippersOpen(mjModel *model, mjData *data)
{
    const char *finger_joints[] = {
        "panda1_finger_joint1", "panda1_finger_joint2",
        "panda2_finger_joint1", "panda2_finger_joint2",
        "panda3_finger_joint1", "panda3_finger_joint2",
        "panda4_finger_joint1", "panda4_finger_joint2"};

    const char *finger_actuators[] = {
        "panda1_actuator8",
        "panda2_actuator8",
        "panda3_actuator8",
        "panda4_actuator8"};

    const double open_pos = 0.04; // joint range is 0..0.04, so max is fully open

    for (const char *joint_name : finger_joints)
    {
        int j_id = mj_name2id(model, mjOBJ_JOINT, joint_name);
        if (j_id >= 0)
        {
            int qpos_adr = model->jnt_qposadr[j_id];
            data->qpos[qpos_adr] = open_pos;
        }
    }

    for (const char *act_name : finger_actuators)
    {
        int a_id = mj_name2id(model, mjOBJ_ACTUATOR, act_name);
        if (a_id >= 0)
        {
            double max_ctrl = model->actuator_ctrlrange[2 * a_id + 1];
            data->ctrl[a_id] = max_ctrl;
        }
    }
}

void setArmsStartPose(mjModel *model, mjData *data, const vector<double> &start_pose)
{
    const char *arm_joints[][7] = {
        {"panda1_joint1", "panda1_joint2", "panda1_joint3", "panda1_joint4", "panda1_joint5", "panda1_joint6", "panda1_joint7"},
        {"panda2_joint1", "panda2_joint2", "panda2_joint3", "panda2_joint4", "panda2_joint5", "panda2_joint6", "panda2_joint7"},
        {"panda3_joint1", "panda3_joint2", "panda3_joint3", "panda3_joint4", "panda3_joint5", "panda3_joint6", "panda3_joint7"},
        {"panda4_joint1", "panda4_joint2", "panda4_joint3", "panda4_joint4", "panda4_joint5", "panda4_joint6", "panda4_joint7"}};

    for (const auto &joint_list : arm_joints)
    {
        for (int i = 0; i < 7; ++i)
        {
            int j_id = mj_name2id(model, mjOBJ_JOINT, joint_list[i]);
            if (j_id >= 0)
            {
                int qpos_adr = model->jnt_qposadr[j_id];
                data->qpos[qpos_adr] = start_pose[i];
            }
        }
    }
}

bool hasCollision(const mjModel *model, const mjData *data, bool print_collisions)
{
    for (int i = 0; i < data->ncon; ++i)
    {
        const mjContact &c = data->contact[i];
        if (c.dist < 0.0)
        {
            if (print_collisions)
            {
                const char *g1 = mj_id2name(model, mjOBJ_GEOM, c.geom1);
                const char *g2 = mj_id2name(model, mjOBJ_GEOM, c.geom2);
                printf("Collision: %s vs %s (penetration %.5f)\n", g1 ? g1 : "geom1", g2 ? g2 : "geom2", c.dist);
            }
            return true;
        }
    }
    return false;
}

vector<Node *> linearInterpolation(const Node *start, const Node *end, int steps, double dt)
{
    int num_actuators = start->q.size();
    int dof = start->q[0].size();

    vector<Node *> trajectory(steps + 1);

    for (int s = 0; s <= steps; s++)
    {
        Node *waypoint = new Node(vector<vector<double>>(num_actuators, vector<double>(dof)), start->t + s * dt);
        trajectory[s] = waypoint;
    }

    for (int arm = 0; arm < num_actuators; arm++) // for each actuator
    {
        auto start_q = start->q[arm];
        auto end_q = end->q[arm];

        int dof = start_q.size();

        vector<double> dq(dof, 0.0);

        for (int i = 0; i < dof; i++)
        {
            dq[i] = (end_q[i] - start_q[i]) / static_cast<double>(steps);
        }

        for (int s = 0; s <= steps; s++)
        {
            vector<double> q(dof, 0);
            for (int i = 0; i < dof; i++)
            {
                q[i] = start_q[i] + dq[i] * s;
            }
            trajectory[s]->q[arm] = q;
        }
    }
    return trajectory;
}

// Output List of Nodes, dim: (dense plan length)
vector<Node *> densifyPlan(const vector<Node *> &waypoints, double dt_sim)
{
    if (waypoints.size() == 0)
    {
        return {};
    }

    int num_actuators = waypoints[0]->q.size();
    vector<Node *> interpolated_plan;

    int M = waypoints.size();
    if (M == 1)
    {
        return waypoints;
    }

    for (int s = 0; s < M - 1; s++)
    {
        const auto &waypoint0 = waypoints[s];
        const auto &waypoint1 = waypoints[s + 1];

        double dt_waypoints = waypoint1->t - waypoint0->t;

        // assumption is that plan does not violate dq limits
        int steps = static_cast<int>(max(1.0, ceil(dt_waypoints / dt_sim)));

        double dt = dt_waypoints / static_cast<double>(steps); // dt between steps

        auto segment = linearInterpolation(waypoint0, waypoint1, steps, dt);
        if (s == 0)
        {
            interpolated_plan.insert(
                interpolated_plan.end(),
                segment.begin(),
                segment.end());
        }
        else
        {
            interpolated_plan.insert(
                interpolated_plan.end(),
                segment.begin() + 1, // skip first
                segment.end());
        }
    }
    return interpolated_plan;
}

// This uses mujocos forward kinematics to compute collisions. If too slow, we can look into
// writing our own collision checker
// Returns->vector of collision pairs, with id's of arms in collision
vector<pair<int, int>> isCollision(const vector<vector<double>> &joint_pos, mjModel *model, mjData *data, const vector<vector<int>> &act_id)
{
    int num_actuators = joint_pos.size();
    int dof = joint_pos[0].size();
    for (int arm = 0; arm < num_actuators; arm++)
    {
        for (int j = 0; j < dof; j++)
        {
            int id = act_id[arm][j];
            data->qpos[id] = joint_pos[arm][j];
        }
    }
    mj_fwdPosition(model, data);
    mj_collision(model, data);
    vector<pair<int, int>> collisions;

    for (int i = 0; i < data->ncon; i++)
    {
        int g1 = data->contact[i].geom1;
        int g2 = data->contact[i].geom2;

        // make sure this is the same as expected somehow
        int b1 = model->geom_bodyid[g1];
        int b2 = model->geom_bodyid[g2];
        collisions.emplace_back(b1, b2);
    }
    return collisions;
}

vector<pair<int, int>> isCollision(const Node *node, mjModel *model, mjData *data, const vector<vector<int>> &act_id)
{
    return isCollision(node->q, model, data, act_id);
}
