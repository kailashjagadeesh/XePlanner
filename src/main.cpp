#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>

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
static const double ready_pose[7] = {0.0, -0.2, 0.0, -2.2, 0.0, 2.0, -2.2};

static void setArmsReadyPose(mjModel *model, mjData *data)
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
                data->qpos[qpos_adr] = ready_pose[i];
            }
        }
    }
}

static void setArmActuatorTargets(mjModel *model, mjData *data)
{
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
                data->ctrl[a_id] = ready_pose[i];
            }
        }
    }
}

static void setGrippersOpen(mjModel *model, mjData *data)
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

static bool hasCollision(const mjModel *model, const mjData *data)
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
    printf("MuJoCo version: %d\n", mj_version());
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
    setArmsReadyPose(m, d);
    setArmActuatorTargets(m, d);
    setGrippersOpen(m, d);
    mj_forward(m, d);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while (d->time - simstart < 1.0 / 60.0)
        {
            mj_step(m, d);
        }

        bool collision = hasCollision(m, d);
        if (!print_collisions && collision && !last_collision)
        {
            printf("Collision detected\n");
        }
        else if (!print_collisions && !collision && last_collision)
        {
            printf("Collision cleared\n");
        }
        last_collision = collision;

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
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
