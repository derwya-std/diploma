#include <memory>
#include <string>
#include <csignal>
#include <rclcpp/rclcpp.hpp>
#include "System.h"
#include "mono-inertial-node.hpp"

using std::string;

static ORB_SLAM3::System* gSLAM = nullptr;

void signal_handler(int) {
    if (gSLAM) {
        RCLCPP_INFO(rclcpp::get_logger("mono_inertial"),
                    "Shutting down ORB-SLAM3 (SIGINT)...");
        gSLAM->Shutdown();
        gSLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    }
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "\nUsage: ros2 run orbslam3 mono-inertial "
                  << "PATH_TO_VOCAB PATH_TO_YAML\n\n";
        return EXIT_FAILURE;
    }
    const string vocab_file = argv[1];
    const string settings_file = argv[2];

    rclcpp::init(argc, argv);

    auto slam = std::make_unique<ORB_SLAM3::System>(
        vocab_file, settings_file,
        ORB_SLAM3::System::IMU_MONOCULAR,
        true);

    gSLAM = slam.get();
    std::signal(SIGINT, signal_handler);

    auto node = std::make_shared<MonocularInertialNode>(
        slam.get(), settings_file, "");

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    RCLCPP_INFO(node->get_logger(), "▶️  mono-inertial node started");
    executor.spin();

    RCLCPP_INFO(node->get_logger(), "Stopping ORB-SLAM3 and saving trajectory…");
    slam->Shutdown();
    slam->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    rclcpp::shutdown();
    return EXIT_SUCCESS;
}