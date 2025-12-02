#include <mujoco/mujoco.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.h"

namespace
{
struct JointLimit
{
    double min;
    double max;
};

double flattenDistance(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        for (size_t j = 0; j < a[i].size(); ++j)
        {
            double d = a[i][j] - b[i][j];
            sum += d * d;
        }
    }
    return std::sqrt(sum);
}

std::vector<std::vector<double>> sampleState(const std::vector<std::vector<JointLimit>> &limits, std::mt19937 &rng)
{
    std::vector<std::vector<double>> q(limits.size());
    for (size_t arm = 0; arm < limits.size(); ++arm)
    {
        q[arm].resize(limits[arm].size());
        for (size_t j = 0; j < limits[arm].size(); ++j)
        {
            std::uniform_real_distribution<double> dist(limits[arm][j].min, limits[arm][j].max);
            q[arm][j] = dist(rng);
        }
    }
    return q;
}

std::vector<std::vector<double>> sampleValidState(mjModel *model,
                                                  mjData *data,
                                                  const std::vector<std::vector<JointLimit>> &limits,
                                                  std::mt19937 &rng,
                                                  int max_attempts,
                                                  const std::vector<std::vector<double>> *avoid = nullptr,
                                                  double min_distance = 0.3)
{
    for (int attempt = 0; attempt < max_attempts; ++attempt)
    {
        auto q = sampleState(limits, rng);

        if (avoid && flattenDistance(q, *avoid) < min_distance)
        {
            continue;
        }

        if (isStateValid(model, data, q))
        {
            return q;
        }
    }

    throw std::runtime_error("Failed to find a collision-free configuration after many attempts.");
}

std::vector<std::vector<JointLimit>> loadJointLimits(const mjModel *model, int num_arms, int dof)
{
    const double fallback_min = -2.8;
    const double fallback_max = 2.8;

    std::vector<std::vector<JointLimit>> limits(num_arms, std::vector<JointLimit>(dof));
    for (int arm = 0; arm < num_arms; ++arm)
    {
        for (int j = 0; j < dof; ++j)
        {
            char name[64];
            std::snprintf(name, sizeof(name), "panda%d_joint%d", arm + 1, j + 1);
            int joint_id = mj_name2id(model, mjOBJ_JOINT, name);
            if (joint_id < 0)
            {
                throw std::runtime_error(std::string("Could not find joint: ") + name);
            }

            if (model->jnt_limited[joint_id])
            {
                limits[arm][j].min = model->jnt_range[2 * joint_id];
                limits[arm][j].max = model->jnt_range[2 * joint_id + 1];
            }
            else
            {
                limits[arm][j].min = fallback_min;
                limits[arm][j].max = fallback_max;
            }
        }
    }

    return limits;
}

std::string serializeToJson(const std::vector<std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>> &samples)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(5);
    oss << "[\n";
    for (size_t i = 0; i < samples.size(); ++i)
    {
        oss << "  {\n";
        oss << "    \"start\": [\n";
        for (size_t arm = 0; arm < samples[i].first.size(); ++arm)
        {
            oss << "      [";
            const auto &angles = samples[i].first[arm];
            for (size_t j = 0; j < angles.size(); ++j)
            {
                oss << angles[j];
                if (j + 1 < angles.size())
                    oss << ", ";
            }
            oss << "]";
            if (arm + 1 < samples[i].first.size())
                oss << ",";
            oss << "\n";
        }
        oss << "    ],\n";
        oss << "    \"goal\": [\n";
        for (size_t arm = 0; arm < samples[i].second.size(); ++arm)
        {
            oss << "      [";
            const auto &angles = samples[i].second[arm];
            for (size_t j = 0; j < angles.size(); ++j)
            {
                oss << angles[j];
                if (j + 1 < angles.size())
                    oss << ", ";
            }
            oss << "]";
            if (arm + 1 < samples[i].second.size())
                oss << ",";
            oss << "\n";
        }
        oss << "    ]\n";
        oss << "  }";
        if (i + 1 < samples.size())
            oss << ",";
        oss << "\n";
    }
    oss << "]\n";
    return oss.str();
}
} // namespace

int main(int argc, char **argv)
{
    const int num_arms = 4;
    const int dof = 7;
    const int max_attempts = 5000;
    const double min_start_goal_distance = 2.0;

    int num_pairs = 20;
    std::string output_path = "random_configs.json";
    unsigned int seed = std::random_device{}();

    if (argc > 1)
    {
        num_pairs = std::stoi(argv[1]);
    }
    if (argc > 2)
    {
        output_path = argv[2];
    }
    if (argc > 3)
    {
        seed = static_cast<unsigned int>(std::stoul(argv[3]));
    }

    if (num_pairs <= 0)
    {
        std::cerr << "Number of pairs must be positive.\n";
        return 1;
    }

    std::mt19937 rng(seed);

    char error[1000] = "Could Not Load Scene";
    mjModel *model = mj_loadXML("franka_emika_panda/scene.xml", nullptr, error, sizeof(error));
    if (!model)
    {
        mju_error("Load model error: %s", error);
    }
    mjData *data = mj_makeData(model);

    auto limits = loadJointLimits(model, num_arms, dof);

    std::vector<std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>> samples;
    samples.reserve(num_pairs);

    for (int i = 0; i < num_pairs; ++i)
    {
        auto start = sampleValidState(model, data, limits, rng, max_attempts);
        auto goal = sampleValidState(model, data, limits, rng, max_attempts, &start, min_start_goal_distance);
        samples.emplace_back(std::move(start), std::move(goal));
    }

    std::ofstream out(output_path);
    if (!out)
    {
        std::cerr << "Failed to open output file: " << output_path << "\n";
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    out << serializeToJson(samples);
    out.close();

    std::cout << "Generated " << samples.size() << " start/goal pairs.\n";
    std::cout << "Seed: " << seed << "\n";
    std::cout << "Saved to: " << output_path << "\n";

    mj_deleteData(data);
    mj_deleteModel(model);
    return 0;
}
