#pragma once
/**
 *  mono-inertial-node.hpp
 *  ──────────────────────
 *  Обгортка ORB-SLAM3 (режим «MONOCULAR + IMU») для ROS 2.
 *
 *  Ключова ідея: карта формується один раз ― від зльоту до посадки,
 *  без жодних «скидань» (reset). Тому у класі відсутня будь-яка
 *  логіка повторної ініціалізації або очищення карти.
 */

#include <queue>
#include <mutex>
#include <string>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <sophus/se3.hpp>
#include "System.h"   // ORB-SLAM3 core

using ImageMsg = sensor_msgs::msg::Image;
using ImuMsg   = sensor_msgs::msg::Imu;
using std::string;

class MonocularInertialNode : public rclcpp::Node
{
public:
    /**
     *  Конструктор.
     *
     *  SLAM     – створений екземпляр ORB_SLAM3::System
     *  settings – шлях до YAML-конфігурації
     *  equal    – зарезервований рядок (залишено для сумісності)
     */
    MonocularInertialNode(ORB_SLAM3::System* SLAM,
                          const string&       settings,
                          const string&       equal);
    ~MonocularInertialNode() override;

private:
    /* ---------------------- Call-back’и ROS ----------------------- */
    void GrabImu(const ImuMsg::SharedPtr  msg);
    void GrabImage(const ImageMsg::SharedPtr msg);

    /* ---------------------------- Utils --------------------------- */
    cv::Mat GetImage(const ImageMsg::SharedPtr msg);
    void    SyncWithImu();                               // головний цикл обробки
    bool    IsTrackingGood(const Sophus::SE3f& pose);    // базова перевірка валідності
    void    ClearOldBuffers(double current_time);        // обрізання черг
    bool    CheckIMUQuality(const std::vector<ORB_SLAM3::IMU::Point>& vImu);

    /* --------------------------- Members -------------------------- */
    ORB_SLAM3::System* SLAM_;                // вказівник, не володіємо пам’яттю

    /* CLAHE (опційне покращення контрасту) */
    bool                     bClahe_ = false;
    cv::Ptr<cv::CLAHE>       clahe_;

    /* Буфери повідомлень */
    std::queue<ImageMsg::SharedPtr> imgBuf_;
    std::queue<ImuMsg::SharedPtr>   imuBuf_;
    std::mutex                      bufMutexImg_;
    std::mutex                      bufMutexImu_;

    /* Таймер для періодичної синхронізації потоків */
    rclcpp::TimerBase::SharedPtr    sync_timer_;

    /* Стан трекінгу */
    bool   mbTrackingLost      = false;
    bool   mbSystemInitialized = false;
    int    mnLostFrames        = 0;

    /* Константи (параметри без «reset» логіки) */
    static constexpr int    MAX_LOST_FRAMES   = 20;   // допустима кількість поспіль «поганих» кадрів
    static constexpr int    MAX_BUFFER_SIZE   = 30;   // max розмір черг
    static constexpr double MAX_TIME_DIFF     = 0.05; // [c] різниця img-imu
    static constexpr double MIN_IMU_RATE      = 5.0;  // [Hz]
    static constexpr size_t MIN_IMU_SAMPLES   = 1;    // мінімум IMU вимірювань

    /* ROS-комунікації */
    rclcpp::Subscription<ImageMsg>::SharedPtr subImg_;
    rclcpp::Subscription<ImuMsg>::SharedPtr   subImu_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr            path_pub_;

    nav_msgs::msg::Path  path_msg_;           // накопичена траєкторія

    /* Статистика */
    size_t mProcessedFrames  = 0;
    size_t mSuccessfulFrames = 0;
};