#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include "utils.h"
#include "rrt_connect.h"

using namespace std;

// --- Global Visualization Variables ---
mjModel *m = NULL;
mjData *d = NULL;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

// Mouse interaction
bool mouse_left = false;
bool mouse_right = false;
bool mouse_middle = false;
double lastx = 0, lasty = 0;

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) mouse_left = action == GLFW_PRESS;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) mouse_right = action == GLFW_PRESS;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) mouse_middle = action == GLFW_PRESS;
    glfwGetCursorPos(window, &lastx, &lasty);
}

void mouse_move_callback(GLFWwindow *window, double xpos, double ypos)
{
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos; lasty = ypos;

    if (!mouse_left && !mouse_right && !mouse_middle) return;

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    
    if (mouse_left) mjv_moveCamera(m, mjMOUSE_ROTATE_H, dx/height, dy/height, &scn, &cam);
    if (mouse_right) mjv_moveCamera(m, mjMOUSE_MOVE_H, dx/height, dy/height, &scn, &cam);
    if (mouse_middle) mjv_moveCamera(m, mjMOUSE_ZOOM, dx/height, dy/height, &scn, &cam);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

int main()
{
    // 1. Load Model
    char error[1000] = "Could not load model";
    m = mj_loadXML("../franka_emika_panda/scene.xml", 0, error, 1000);
    if (!m) mju_error("Load error: %s", error);
    d = mj_makeData(m);

    // 2. Init Visualization
    if (!glfwInit()) mju_error("Could not init GLFW");
    GLFWwindow *window = glfwCreateWindow(1200, 900, "RRT-Connect Single Arm", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_move_callback);
    glfwSetScrollCallback(window, scroll_callback);

    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // 3. Define Task
    // Safe Home Pose

    std::vector<double> start_q = {
        0.0,
        -0.785398, // -45 degrees
        0.0,
        -2.35619, // -135 degrees
        0.0,
        1.57,    // 90 degrees
        0.785398 // 45 degrees
    };

    std::vector<double> goal_q = {
        0.0,
        -0.5,
        0.0,
        -1.8,
        0.0,
        2.3,
        0.8};

    // Set Environment to Start
    vector<vector<double>> start_multi(4, vector<double>(7, 0.0));
    start_multi[0] = start_q; 
    setAllArmsQpos(m, d, start_multi);
    mj_forward(m, d);

    // 4. Run Planner
    int agent_id = 0; // Plan for Agent 0
    int num_agents = 1; // Total agents in scene
    int dof = 7;

    cout << "=== Running RRT-Connect ===" << endl;
    RRTConnectPlanner planner(m, agent_id, dof, num_agents);
    
    vector<Node*> waypoints = planner.plan(start_q, goal_q);

    if (waypoints.empty()) {
        cout << "Planning failed!" << endl;
        return 0;
    }

    // 5. Densify Path for Playback
    double dt_playback = 0.01;
    vector<Node*> trajectory = densifyPlan(waypoints, dt_playback);

    // 6. Playback Loop
    while (!glfwWindowShouldClose(window))
    {
        double time = d->time;
        int step = min((int)(time / dt_playback), (int)trajectory.size() - 1);

        // Control Agent 0 based on plan
        vector<double> current_q = trajectory[step]->q[0];
        
        // Map to control inputs
        for(int j=0; j<7; ++j) {
            int id = mj_name2id(m, mjOBJ_ACTUATOR, ("panda1_actuator" + to_string(j+1)).c_str());
            if(id != -1) d->ctrl[id] = current_q[j];
        }

        mj_step(m, d);

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    mj_deleteData(d);
    mj_deleteModel(m);
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    glfwTerminate();

    return 0;
}