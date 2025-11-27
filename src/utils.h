#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <cmath>
#include <cstring>

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
bool hasCollision(const mjModel *model, const mjData *data, bool print_collisions);

void setAllArmsQpos(const mjModel* model, mjData* data, const std::vector<std::vector<double>>& q_multi);
bool isStateValid(const mjModel *model, mjData *data, const std::vector<std::vector<double>> &q_multi, bool print_collisions = false);
bool isEdgeValid(const mjModel *model, mjData *data, const std::vector<std::vector<double>> &q_from, const std::vector<std::vector<double>> &q_to, int num_substeps, bool print_collisions = false);

vector<Node *> linearInterpolation(const Node *start, const Node *end, int steps, double dt);

vector<Node *> densifyPlan(const vector<Node *> &waypoints, double dt_sim);

int geomToAgent(const mjModel* model, int geom_id, int num_agents);
