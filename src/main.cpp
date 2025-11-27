#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include "utils.h"
#include "rrt_connect.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

using namespace std;

mjModel *m = NULL; // MuJoCo model
mjData *d = NULL;  // MuJoCo data
mjvCamera cam;     // abstract camera
mjvOption opt;     // visualization options
mjvScene scn;      // abstract scene
mjrContext con;    // custom GPU context

static bool mouse_left = false;
static bool mouse_right = false;
static bool mouse_middle = false;
static double lastx = 0.0;
static double lasty = 0.0;
static bool print_collisions = false;
static bool last_collision = false;

static const vector<double> start_pose = {0.0, -0.2, 0.0, -2.2, 0.0, 2.0, -2.2};

static const int num_actuators = 4;
static const double dt = 0.01;
static const double dq_max = 3.142; // rad / s

static double envOrDefault(const char *name, double fallback)
{
    const char *val = std::getenv(name);
    if (!val || !*val)
    {
        return fallback;
    }

    char *end = nullptr;
    double parsed = std::strtod(val, &end);
    return (end == val) ? fallback : parsed;
}

static void configureCameraForModel(const mjModel *model, mjvCamera *camera)
{
    const double extent = mju_max(model->stat.extent, 1e-3);

    camera->lookat[0] = model->stat.center[0];
    camera->lookat[1] = model->stat.center[1];
    camera->lookat[2] = model->stat.center[2];

    camera->distance = envOrDefault("MJ_CAM_DISTANCE", 4.0 * extent);
    camera->azimuth = envOrDefault("MJ_CAM_AZIMUTH", 90.0);
    camera->elevation = envOrDefault("MJ_CAM_ELEVATION", -45.0);
}

static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    (void)mods;

    if (button == GLFW_MOUSE_BUTTON_LEFT)
        mouse_left = (action == GLFW_PRESS);
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        mouse_middle = (action == GLFW_PRESS);
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        mouse_right = (action == GLFW_PRESS);

    if (mouse_left || mouse_middle || mouse_right)
        glfwGetCursorPos(window, &lastx, &lasty);
}

static void mouse_move_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (!mouse_left && !mouse_middle && !mouse_right)
        return;

    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    int width = 1, height = 1;
    glfwGetWindowSize(window, &width, &height);
    double scale = (height > 0) ? 1.0 / height : 1.0;

    if (mouse_left)
        mjv_moveCamera(m, mjMOUSE_ROTATE_H, scale * dx, scale * dy, &scn, &cam);
    else if (mouse_right)
        mjv_moveCamera(m, mjMOUSE_MOVE_H, scale * dx, scale * dy, &scn, &cam);
    else if (mouse_middle)
        mjv_moveCamera(m, mjMOUSE_ZOOM, scale * dx, scale * dy, &scn, &cam);
}

static void scroll_callback(GLFWwindow * /*window*/, double /*xoffset*/, double yoffset)
{
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, &scn, &cam);
}

int main()
{
    char error[1000] = "Could Not Load Scene";
    m = mj_loadXML("../franka_emika_panda/scene.xml", nullptr, error, sizeof(error));

    if (!m)
    {
        mju_error("Load model error: %s", error);
    }

    d = mj_makeData(m);

    // init GLFW
    if (!glfwInit())
    {
        mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    GLFWwindow *window = glfwCreateWindow(1200, 1200, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_move_callback);
    glfwSetScrollCallback(window, scroll_callback);

    print_collisions = (std::getenv("MJ_PRINT_COLLISIONS") != nullptr);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    configureCameraForModel(m, &cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // set initial arm pose and gripper state
    setArmsStartPose(m, d, start_pose);
    setArmActuatorTargets(m, d, start_pose);
    setGrippersOpen(m, d);
    mj_forward(m, d);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    vector<double> end_pose = {0.0, -2, 0.0, -1.2, 0.5, 1.0, -1};

    // Build start/goal nodes for the planner (same pose applied to all arms).
    vector<vector<double>> start_poses(num_actuators, start_pose);
    vector<vector<double>> end_poses(num_actuators, end_pose);
    Node *start = new Node(start_poses, 0.0);
    Node *goal = new Node(end_poses, 0.0);

    // ----------------- THIS IS WHERE PLANNER FUNCTION CALL SHOULD GO ----------------------- //
    printf("Planning with RRT-Connect...\n");
    // Run a simple RRT-Connect to get sparse waypoints, then densify them.
    vector<Node *> plan = rrtConnect(
        m,
        num_actuators,
        static_cast<int>(start_pose.size()),
        start,
        goal,
        20000, //max_iters
        0.6);//step_size
    printf("Planning done. Sparse waypoints: %zu\n", plan.size());

    auto dense_plan = densifyPlan(plan, dt); // this function will densify the plan with linear interpolation to ensure that desired timesteps are followed
    printf("Dense trajectory steps: %zu\n", dense_plan.size());


    int dof = dense_plan[0]->q[0].size();

    // map arm, joint to location to control data index
    vector<vector<int>> act_id(num_actuators, vector<int>(dof, -1));
    for (int arm = 0; arm < num_actuators; ++arm) {
        for (int j = 0; j < dof; ++j) {
            char name[64];
            snprintf(name, sizeof(name), "panda%d_actuator%d", arm+1, j+1);
            int id = mj_name2id(m, mjOBJ_ACTUATOR, name); // returns -1 if not found
            if (id == -1) throw runtime_error("Error: Could not find actuator id");
            act_id[arm][j] = id;
        }
    }
    
    auto body_to_arm = bodyToArm(act_id, m, num_actuators, dof);

    while (!glfwWindowShouldClose(window))
    {
        mj_step1(m, d);
        double t_sim = d->time;

        int t = min(static_cast<int>(t_sim / dt), static_cast<int>(dense_plan.size() - 1));
        Node* curr_node = dense_plan[t];
        for (int arm = 0; arm < curr_node->q.size(); arm++)
        {
            for (int j = 0; j < dof; j++)
            {
                int id = act_id[arm][j];
                d->ctrl[id] = curr_node->q[arm][j];
            }
        }
        mj_step2(m, d);

        bool collision = hasCollision(m, d, print_collisions);
        if (!print_collisions && collision && !last_collision)
        {
            printf("Collision detected\n");
        }

        last_collision = collision;

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
