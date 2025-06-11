#include "mono-inertial-node.hpp"

MonocularInertialNode::MonocularInertialNode(ORB_SLAM3::System* SLAM,
                                             const string&      settings,
                                             const string&      /*equal*/)
: Node("mono_inertial_node"), SLAM_(SLAM) {
    string img_topic = this->declare_parameter<string>("image_topic", "/camera/image");
    string imu_topic = this->declare_parameter<string>("imu_topic", "/imu");

    bClahe_ = this->declare_parameter<bool>("use_clahe", false);
    if (bClahe_)
        clahe_ = cv::createCLAHE(4.0, cv::Size(8, 8));

    rclcpp::SensorDataQoS qos;

    subImg_ = create_subscription<ImageMsg>(
        img_topic, qos,
        std::bind(&MonocularInertialNode::GrabImage, this, std::placeholders::_1));

    subImu_ = create_subscription<ImuMsg>(
        imu_topic, qos,
        std::bind(&MonocularInertialNode::GrabImu, this, std::placeholders::_1));

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("orbslam/pose", 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("orbslam/path", 10);

    sync_timer_ = create_wall_timer(
        std::chrono::milliseconds(5),
        std::bind(&MonocularInertialNode::SyncWithImu, this));

    RCLCPP_INFO(get_logger(), "Mono-Inertial ORB-SLAM3 node initialised");
}

MonocularInertialNode::~MonocularInertialNode() {
    if (SLAM_) SLAM_->Shutdown();
}

void MonocularInertialNode::GrabImu(const ImuMsg::SharedPtr msg) {
    std::scoped_lock lock(bufMutexImu_);
    imuBuf_.push(msg);
}

void MonocularInertialNode::GrabImage(const ImageMsg::SharedPtr msg) {
    std::scoped_lock lock(bufMutexImg_);
    imgBuf_.push(msg);
}

void MonocularInertialNode::SyncWithImu() {
    ImageMsg::SharedPtr img_msg;
    {
        std::scoped_lock lock(bufMutexImg_);
        if (imgBuf_.empty()) { return; }
        img_msg = imgBuf_.front();
        imgBuf_.pop();
    }

    double t_img = img_msg->header.stamp.sec + 1e-9 * img_msg->header.stamp.nanosec;

    std::vector<ORB_SLAM3::IMU::Point> vImu;
    {
        std::scoped_lock lock(bufMutexImu_);
        while (!imuBuf_.empty()) {
            auto& imu = imuBuf_.front();
            double t_imu = imu->header.stamp.sec + 1e-9 * imu->header.stamp.nanosec;

            if (t_imu <= t_img) {
                cv::Point3f acc_pt(imu->linear_acceleration.x,
                                imu->linear_acceleration.y,
                                imu->linear_acceleration.z);
                cv::Point3f gyr_pt(imu->angular_velocity.x,
                                imu->angular_velocity.y,
                                imu->angular_velocity.z);
                vImu.emplace_back(acc_pt, gyr_pt, t_imu);
                imuBuf_.pop();
            } else break;
        }
    }

    if (!CheckIMUQuality(vImu))
        return;

    cv::Mat img = GetImage(img_msg);
    if (img.empty()) return;

    Sophus::SE3f Tcw = SLAM_->TrackMonocular(img, t_img, vImu);
    ++mProcessedFrames;

    bool good = IsTrackingGood(Tcw);
    if (good) ++mSuccessfulFrames;

    if (good) {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = img_msg->header;
        pose_msg.pose.position.x =  Tcw.translation().x();
        pose_msg.pose.position.y =  Tcw.translation().y();
        pose_msg.pose.position.z =  Tcw.translation().z();

        Eigen::Quaternionf q(Tcw.rotationMatrix());
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();

        pose_pub_->publish(pose_msg);

        path_msg_.header = pose_msg.header;
        path_msg_.poses.push_back(pose_msg);
        path_pub_->publish(path_msg_);
    }

    ClearOldBuffers(t_img);
}

cv::Mat MonocularInertialNode::GetImage(const ImageMsg::SharedPtr msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        return {};
    }

    cv::Mat img = cv_ptr->image.clone();
    if (bClahe_) clahe_->apply(img, img);
    return img;
}

bool MonocularInertialNode::IsTrackingGood(const Sophus::SE3f& pose) {
    auto data = pose.matrix();
    return data.allFinite();
}

void MonocularInertialNode::ClearOldBuffers(double /*current_time*/) {
    {
        std::scoped_lock lock(bufMutexImg_);
        while (imgBuf_.size() > MAX_BUFFER_SIZE) imgBuf_.pop();
    }
    {
        std::scoped_lock lock(bufMutexImu_);
        while (imuBuf_.size() > MAX_BUFFER_SIZE) imuBuf_.pop();
    }
}

bool MonocularInertialNode::CheckIMUQuality(const std::vector<ORB_SLAM3::IMU::Point>& vImu) {
    if (vImu.size() < MIN_IMU_SAMPLES) return false;

    double t0 = vImu.front().t;
    double t1 = vImu.back().t;
    double rate = static_cast<double>(vImu.size()) / std::max(t1 - t0, 1e-6);
    return rate >= MIN_IMU_RATE;
}