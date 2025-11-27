#pragma once
#include <mujoco/mujoco.h>
#include <vector>
#include <cmath>
#include <regex>

using namespace std;

struct Node
{
	vector<vector<double>> q;
	double t;
	Node() {}
	Node(const vector<vector<double>> &q_, double t_) : q(q_), t(t_) {}
};

void setArmActuatorTargets(mjModel *model, mjData *data, const vector<double> &target_pose);
void setGrippersOpen(mjModel *model, mjData *data);
void setArmsStartPose(mjModel *model, mjData *data, const vector<double> &start_pose);
bool hasCollision(const mjModel *model, const mjData *data, bool print_collisions); // used to check collisions during sim execution

vector<Node *> linearInterpolation(const Node *start, const Node *end, int steps, double dt);
vector<Node *> densifyPlan(const vector<Node *> &waypoints, double dt_sim);

bool isSingleArmCollision(const vector<double> &joint_pos, mjModel *model, mjData *data, int arm, const vector<vector<int>> &joint_id, const vector<int> &body_to_arm);
vector<pair<int, int>> isMultiArmCollision(const vector<vector<double>> &joint_pos, mjModel *model, mjData *data, const vector<vector<int>> &joint_id, const vector<int> &body_to_arm);
vector<pair<int, int>> isMultiArmCollision(const Node *node, mjModel *model, mjData *data, const vector<vector<int>> &joint_id, const vector<int> &body_to_arm);
vector<vector<int>> buildBodyChildren(const mjModel *model);
vector<int> bodyToArm(const vector<vector<int>> &act_id, const mjModel *model, int num_actuators, int dof);