#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include "utils.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

#include "cbs.h"

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

// static const vector<double> start_pose = {0.0, -0.2, 0.0, -2.2, 0.0, 2.0, -2.2};
static const vector<double> start_pose = {0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0};

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

    std::vector<double> HOME_POSE = {
        0.0,
        -0.785398, // -45 degrees
        0.0,
        -2.35619, // -135 degrees
        0.0,
        1.57,    // 90 degrees
        0.785398 // 45 degrees
    };

    std::vector<double> UPRIGHT_POSE = {
        0.0,
        -0.5,
        0.0,
        -1.8,
        0.0,
        2.3,
        0.8};

    std::vector<double> EASY_POSE = {0.0, -0.785398, 0.0, -2.35619, 0.0, 1.57, 0.785398};

    // ----------------- CBS PLANNER SETUP ----------------- //
    std::cout << "\n===== RUNNING FULL CBS PLANNER =====\n";

    int num_agents = 1;
    int dofs = start_pose.size();

    CBSPlanner planner(m, dq_max, dt, num_agents, dofs);

    std::vector<std::vector<double>> start_poses(num_agents, EASY_POSE);

    // std::vector<double> end_pose = {0.0, -2.0, 0.0, -1.2, 0.5, 1.0, -1.0};
    // std::vector<double> end_pose = {1.0, -0.5, 0.0, -1.0, 0.0, 1.0, 0.0};
    std::vector<std::vector<double>> goal_poses(num_agents, EASY_POSE);
    // goal_poses[0][6] += 0.01;

    std::cout << "\n[DEBUG] Checking State Validity:\n";

    // For single arm, check Agent 0 (index 0)
    std::vector<std::vector<double>> start_q_multi(num_agents, EASY_POSE);
    std::vector<std::vector<double>> goal_q_multi(num_agents, EASY_POSE);

    for (int a = 1; a < num_agents; ++a)
    {
        start_q_multi[a] = std::vector<double>(dofs, 0.0);
        goal_q_multi[a] = std::vector<double>(dofs, 0.0);
    }

    // Check start pose validity
    bool start_valid = isStateValid(m, d, start_q_multi, true);
    std::cout << "Start Pose is Valid: " << (start_valid ? "YES" : "NO") << "\n";

    // Check goal pose validity
    bool goal_valid = isStateValid(m, d, goal_q_multi, true);
    std::cout << "Goal Pose is Valid: " << (goal_valid ? "YES" : "NO") << "\n";

    std::vector<Node *> plan = planner.plan(start_poses, goal_poses);
    if (plan.empty())
    {
        std::cerr << "[MAIN] CBS failed to find a plan.\n";
        // clean up and exit
        mjv_freeScene(&scn);
        mjr_freeContext(&con);
        mj_deleteData(d);
        mj_deleteModel(m);
        return 0;
    }

    std::cout << "[MAIN] CBS returned plan with " << plan.size() << " waypoints\n";

    // ----------------- THIS IS WHERE PLANNER FUNCTION CALL SHOULD GO ----------------------- //
    // INPUT: vector<Node*> -> dim(num_waypoints)

    // vector<vector<double>> start_poses(num_actuators, start_pose);
    // vector<vector<double>> end_poses(num_actuators, end_pose);
    // Node *start = new Node(start_poses, 0);
    // Node *mid = new Node(end_poses, 1);
    // Node *end = new Node(start_poses, 2);

    // vector<Node *> plan = {start, mid, end};
    // --------------------------------------------------------------------------------------- //

    auto dense_plan = densifyPlan(plan, dt); // this function will densify the plan with linear interpolation to ensure that desired timesteps are followed

    int dof = dense_plan[0]->q[0].size();

    // map arm, joint to location to control data index
    vector<vector<int>> act_id(num_actuators, vector<int>(dof, -1));
    for (int arm = 0; arm < num_actuators; ++arm)
    {
        for (int j = 0; j < dof; ++j)
        {
            char name[64];
            snprintf(name, sizeof(name), "panda%d_actuator%d", arm + 1, j + 1);
            int id = mj_name2id(m, mjOBJ_ACTUATOR, name); // returns -1 if not found
            if (id == -1)
                throw runtime_error("Error: Could not find actuator id");
            act_id[arm][j] = id;
        }
    }

    while (!glfwWindowShouldClose(window))
    {
        mj_step1(m, d);
        double t_sim = d->time;

        int t = min(static_cast<int>(t_sim / dt), static_cast<int>(dense_plan.size() - 1));
        Node *curr_node = dense_plan[t];
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
