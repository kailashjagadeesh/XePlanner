#include <mujoco/mujoco.h>
#include <vector>
#include <cmath>

using namespace std;

struct Waypoint {
	std::vector<double> q;
	double time;
	Waypoint() : q(), time(0.0) {}
	Waypoint(const std::vector<double> &q_, double t) : q(q_), time(t) {}
};

void setArmActuatorTargets(mjModel *model, mjData *data, const vector<double> &target_pose);
void setGrippersOpen(mjModel *model, mjData *data);
void setArmsStartPose(mjModel *model, mjData *data, const vector<double> &start_pose);
bool hasCollision(const mjModel *model, const mjData *data, bool print_collisions);

vector<Waypoint> linearInterpolation(const Waypoint &start, const Waypoint &end, int steps, double dt);
vector<vector<Waypoint>> densifyPlan(const vector<vector<Waypoint>> &waypoints, double dt_sim);
