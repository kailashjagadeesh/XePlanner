#include <mujoco/mujoco.h>
#include <vector>
#include <cmath>

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

vector<Node*> linearInterpolation(const Node* start, const Node* end, int steps, double dt);

vector<Node*> densifyPlan(const vector<Node*> &waypoints, double dt_sim);
