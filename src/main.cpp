#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cbs.h"
#include "ecbs.h"
#include "rrt_connect.h"
#include "utils.h"

using namespace std;

enum class PlannerType
{
    CBS,
    ECBS,
    RRT
};

static constexpr int kNumArms = 4;

struct SceneConfig
{
    std::string scene_xml;
    std::vector<std::vector<double>> starts; // per-arm if size==num_agents, otherwise broadcast
    std::vector<std::vector<double>> goals;  // per-arm if size==num_agents, otherwise broadcast
};

// -------------- GLOBALS / VISUALIZATION -------------- //
mjModel *m = nullptr; // MuJoCo model
mjData *d = nullptr;  // MuJoCo data
mjvCamera cam;        // abstract camera
mjvOption opt;        // visualization options
mjvScene scn;         // abstract scene
mjrContext con;       // custom GPU context

static bool mouse_left = false;
static bool mouse_right = false;
static bool mouse_middle = false;
static double lastx = 0.0;
static double lasty = 0.0;
static bool print_collisions = false;
static bool last_collision = false;

// --- POSE DEFINITIONS ---
static const vector<double> START_POSE = {-1.0472, -0.2, 0.0, -2.2, 0.0, 2.0, -2.2};
static const vector<double> END_POSE_1 = {0.0349, -0.7090, 0.0314, -2.2983, 0.0035, 1.8429, -2.2000};
static const vector<double> HOME_POSE = {0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785};
static const vector<double> UPRIGHT_POSE = {0.0, 0.0, 0.0, -1.0, 0.0, 2.0, 0.8};
static const vector<double> ALT_POSE = {0.8, -1.5, 0.5, -1.5, 0.0, 2.0, -0.7};

static const int num_actuators = 4;
static const double dt = 0.01;
static const double dq_max = 3.142; // rad / s

// -------------- HELPERS -------------- //
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

static string toLower(const string &s)
{
    string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c)
                   { return std::tolower(c); });
    return out;
}

static string trim(const string &s)
{
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
        ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
        --end;
    return s.substr(start, end - start);
}

static std::vector<double> parseConfigVector(const std::string &line)
{
    std::vector<double> vals;
    auto l = line.find('{');
    auto r = line.find('}', l == std::string::npos ? 0 : l + 1);
    if (l == std::string::npos || r == std::string::npos || r <= l + 1)
        return vals;

    std::string inner = line.substr(l + 1, r - l - 1);
    std::stringstream ss(inner);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        tok = trim(tok);
        if (tok.empty())
            continue;
        try
        {
            vals.push_back(std::stod(tok));
        }
        catch (...)
        {
            // ignore bad tokens
        }
    }
    return vals;
}

static bool startsWith(const std::string &s, const std::string &prefix)
{
    if (s.size() < prefix.size())
        return false;
    return std::equal(prefix.begin(), prefix.end(), s.begin(), [](char a, char b)
                      { return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); });
}

// Try to parse patterns like "start_config_3:" or any start/goal line with an index.
static bool parseIndexedConfig(const std::string &line,
                               const std::string &tag,
                               std::vector<std::vector<double>> &storage)
{
    std::string lower = toLower(line);
    std::string ltag = toLower(tag);

    auto pos = lower.find(ltag);
    if (pos == std::string::npos)
        return false;

    std::size_t idx_pos = lower.find_first_of("0123456789", pos + ltag.size());
    if (idx_pos == std::string::npos)
        return false;

    int idx = std::stoi(lower.substr(idx_pos));
    if (idx < 1 || idx > kNumArms)
        return false;

    auto cfg = parseConfigVector(line);
    if (!cfg.empty())
    {
        if (static_cast<int>(storage.size()) < kNumArms)
            storage.resize(kNumArms);
        storage[idx - 1] = std::move(cfg);
        return true;
    }
    return false;
}

static std::string parseSceneFileArg(int argc, char **argv)
{
    std::string from_env;
    if (const char *env = std::getenv("SCENE_FILE"))
    {
        from_env = env;
    }

    std::string from_cli;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "--scene-file" || arg == "-s") && i + 1 < argc)
        {
            from_cli = argv[i + 1];
            break;
        }
    }

    if (!from_cli.empty())
        return from_cli;
    if (!from_env.empty())
        return from_env;
    return "scenes/scene.txt"; // default
}

static std::filesystem::path resolveSceneFilePath(const std::string &path_hint)
{
    std::filesystem::path p(path_hint);
    if (p.is_absolute())
        return p;

    std::vector<std::filesystem::path> candidates = {
        std::filesystem::current_path() / p,
        std::filesystem::current_path().parent_path() / p};

    for (const auto &c : candidates)
    {
        if (std::filesystem::exists(c))
            return c;
    }
    return std::filesystem::current_path() / p;
}

static std::filesystem::path resolveSceneXmlPath(const std::string &xml_path_str,
                                                 const std::filesystem::path &scene_file_dir)
{
    std::filesystem::path xml(xml_path_str);
    if (xml.is_absolute())
        return xml;

    std::vector<std::filesystem::path> candidates = {
        scene_file_dir / xml,
        std::filesystem::current_path() / xml,
        std::filesystem::current_path().parent_path() / xml};

    if (xml.parent_path().empty())
    {
        candidates.push_back(std::filesystem::current_path().parent_path() / "franka_emika_panda" / xml);
    }

    for (const auto &c : candidates)
    {
        if (std::filesystem::exists(c))
            return c;
    }
    return xml;
}

static SceneConfig loadSceneConfig(const std::filesystem::path &scene_file)
{
    SceneConfig cfg;
    cfg.scene_xml = "franka_emika_panda/scene.xml";
    cfg.starts.assign(kNumArms, {});
    cfg.goals.assign(kNumArms, {});
    std::vector<double> broadcast_start;
    std::vector<double> broadcast_goal;

    std::ifstream in(scene_file);
    if (!in)
    {
        std::cerr << "[MAIN] Could not open scene file " << scene_file << ", using defaults.\n";
        return cfg;
    }

    bool scene_set = false;
    std::string line;
    while (std::getline(in, line))
    {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#')
            continue;

        if (!scene_set)
        {
            cfg.scene_xml = t;
            scene_set = true;
            continue;
        }

        if (startsWith(t, "start"))
        {
            if (!parseIndexedConfig(t, "start_config_", cfg.starts))
            {
                auto v = parseConfigVector(t);
                if (!v.empty())
                    broadcast_start = std::move(v);
            }
        }
        else if (startsWith(t, "goal"))
        {
            if (!parseIndexedConfig(t, "goal_config_", cfg.goals))
            {
                auto v = parseConfigVector(t);
                if (!v.empty())
                    broadcast_goal = std::move(v);
            }
        }
    }

    // Fill missing entries with broadcast or defaults
    for (int i = 0; i < kNumArms; ++i)
    {
        if (cfg.starts[i].empty())
            cfg.starts[i] = !broadcast_start.empty() ? broadcast_start : START_POSE;
    }
    for (int i = 0; i < kNumArms; ++i)
    {
        if (cfg.goals[i].empty())
            cfg.goals[i] = !broadcast_goal.empty() ? broadcast_goal : HOME_POSE;
    }

    return cfg;
}

static PlannerType parsePlanner(int argc, char **argv)
{
    string from_env;
    if (const char *env = std::getenv("PLANNER"))
    {
        from_env = toLower(env);
    }

    string from_cli;
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if ((arg == "--planner" || arg == "-p") && i + 1 < argc)
        {
            from_cli = toLower(argv[i + 1]);
            break;
        }
    }

    string sel = from_cli.empty() ? from_env : from_cli;
    if (sel == "ecbs")
        return PlannerType::ECBS;
    if (sel == "rrt" || sel == "rrt-connect" || sel == "rrtconnect")
        return PlannerType::RRT;
    return PlannerType::CBS; // default
}

// -------------- MAIN -------------- //
int main(int argc, char **argv)
{
    PlannerType planner_type = parsePlanner(argc, argv);

    // Scene selection and configs
    std::string scene_file_hint = parseSceneFileArg(argc, argv);
    std::filesystem::path scene_file_path = resolveSceneFilePath(scene_file_hint);
    SceneConfig scene_cfg = loadSceneConfig(scene_file_path);
    std::filesystem::path resolved_scene_xml = resolveSceneXmlPath(scene_cfg.scene_xml, scene_file_path.parent_path());

    std::cout << "[MAIN] Using scene file: " << scene_file_path << "\n";
    std::cout << "[MAIN] Loading XML: " << resolved_scene_xml << "\n";

    char error[1000] = "Could Not Load Scene";
    m = mj_loadXML(resolved_scene_xml.string().c_str(), nullptr, error, sizeof(error));

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

    
    // ----------------- CONFIGURATION SETUP ----------------- //

    const int num_agents = kNumArms;
    std::vector<std::vector<double>> start_poses = scene_cfg.starts.empty() ? std::vector<std::vector<double>>(num_agents, START_POSE) : scene_cfg.starts;
    std::vector<std::vector<double>> goal_poses = scene_cfg.goals.empty() ? std::vector<std::vector<double>>(num_agents, END_POSE_1) : scene_cfg.goals;

    // Fallback fill for malformed files
    for (int i = 0; i < num_agents; ++i)
    {
        if (start_poses[i].empty())
            start_poses[i] = START_POSE;
        if (goal_poses[i].empty())
            goal_poses[i] = END_POSE_1;
        if (goal_poses[i].size() != start_poses[i].size())
            goal_poses[i] = goal_poses[0];
    }

    const int dofs = static_cast<int>(start_poses[0].size());

    // Log the final start/goal per arm
    std::cout << "[MAIN] Final start/goal configs per arm:\n";
    for (int i = 0; i < num_agents; ++i)
    {
        std::cout << "  Arm " << i << " start: {";
        for (size_t j = 0; j < start_poses[i].size(); ++j)
        {
            std::cout << start_poses[i][j];
            if (j + 1 < start_poses[i].size())
                std::cout << ", ";
        }
        std::cout << "}  goal: {";
        for (size_t j = 0; j < goal_poses[i].size(); ++j)
        {
            std::cout << goal_poses[i][j];
            if (j + 1 < goal_poses[i].size())
                std::cout << ", ";
        }
        std::cout << "}\n";
    }

    // Apply initial start poses to the simulation so visualizer starts correctly
    setAllArmsQpos(m, d, start_poses);
    mj_forward(m, d);

    int dof = start_poses[0].size();

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
    vector<vector<int>> joint_id(num_actuators, vector<int>(dof, -1));
    for (int arm = 0; arm < num_actuators; ++arm)
    {
        for (int j = 0; j < dof; ++j)
        {
            char name[64];
            snprintf(name, sizeof(name), "panda%d_joint%d", arm + 1, j + 1);
            int jid = mj_name2id(m, mjOBJ_JOINT, name);
            if (jid < 0)
                throw runtime_error("Could not find joint");
            joint_id[arm][j] = jid;
        }
    }
    auto body_to_arm = bodyToArm(act_id, m, num_actuators, dof);

    // ----------------- PLANNER DISPATCH ----------------- //
    vector<Node *> plan;
    if (planner_type == PlannerType::CBS || planner_type == PlannerType::ECBS)
    {
        if (planner_type == PlannerType::CBS)
            std::cout << "\n===== RUNNING CBS PLANNER =====\n";
        else
            std::cout << "\n===== RUNNING ECBS PLANNER =====\n";

        auto t_start_plan = std::chrono::high_resolution_clock::now();

        if (planner_type == PlannerType::CBS)
        {
            CBSPlanner planner(m, dq_max, dt, num_agents, dofs, body_to_arm, joint_id);
            plan = planner.plan(start_poses, goal_poses);
        }
        else
        {
            ECBSPlanner planner(m, dq_max, dt, num_agents, dofs, body_to_arm, joint_id, 1.5);
            plan = planner.plan(start_poses, goal_poses);
        }

        auto t_end_plan = std::chrono::high_resolution_clock::now();
        double plan_seconds = std::chrono::duration<double>(t_end_plan - t_start_plan).count();

        if (plan.empty())
        {
            std::cerr << "[MAIN] Planner failed to find a plan.\n";
            plan.push_back(new Node(start_poses, 0.0));
        }
        else
        {
            std::cout << "[MAIN] Planner returned plan with " << plan.size() << " waypoints\n";
            std::cout << "[MAIN] Planning time: " << plan_seconds << " s\n";
        }
    }
    else if (planner_type == PlannerType::RRT)
    {
        std::cout << "\n===== RUNNING RRT-CONNECT =====\n";
        auto t_start_plan = std::chrono::high_resolution_clock::now();
        Node *start = new Node(start_poses, 0.0);
        Node *goal = new Node(goal_poses, 0.0);
        plan = rrtConnect(
            m,
            num_actuators,
            dof,
            start,
            goal,
            20000, // max_iters
            0.6);  // step_size
        auto t_end_plan = std::chrono::high_resolution_clock::now();
        double plan_seconds = std::chrono::duration<double>(t_end_plan - t_start_plan).count();
        if (plan.empty())
        {
            std::cerr << "[MAIN] RRT failed to find a plan.\n";
            plan.push_back(new Node(start_poses, 0.0));
        }
        else
        {
            std::cout << "[MAIN] RRT returned " << plan.size() << " sparse waypoints\n";
            plan = densifyPlan(plan, dt);
            std::cout << "[MAIN] Dense plan has " << plan.size() << " steps\n";
            std::cout << "[MAIN] Planning time: " << plan_seconds << " s\n";
        }
    }

    auto dense_plan = plan; // CBS/ECBS already stepwise; RRT was densified if selected above
    printf("Dense trajectory steps: %zu\n", dense_plan.size());

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

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    setAllArmsQpos(m, d, start_poses);
    mj_forward(m, d);

    while (!glfwWindowShouldClose(window))
    {
        mj_step1(m, d);
        double t_sim = d->time;

        int t = std::min(static_cast<int>(t_sim / dt), static_cast<int>(dense_plan.size() - 1));
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
