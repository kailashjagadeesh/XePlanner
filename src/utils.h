#include <mujoco/mujoco.h>
#include <vector>
#include <cmath>

using namespace std;

void setArmActuatorTargets(mjModel *model, mjData *data, const double target_pose[]);
void setGrippersOpen(mjModel *model, mjData *data);
void setArmsStartPose(mjModel *model, mjData *data, const double start_pose[]);
bool hasCollision(const mjModel *model, const mjData *data, bool print_collisions);

vector<vector<double>> linearInterpolation(const vector<double> &start, const vector<double> &end, int steps);
vector<vector<vector<double>>> densifyPlan(const vector<vector<vector<double>>> &waypoints, double dt, double dq_max);