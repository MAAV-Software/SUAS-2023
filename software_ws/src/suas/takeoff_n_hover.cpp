#include <ros/ros.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/CommandTOL.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/CommandTriggerControl.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>

#define FLIGHT_ALTITUDE 35
#define HOVER_DURATION 120.0
#define HOVER_X 20.683  // Adjust these values according to your desired hover position (currently set to center of air strip)
#define HOVER_Y 54.864

// five midpoints, evenly distributed
// (20.6833, 18.288), (20.6833, 36.576), (20.6833, 54.864), (20.6833, 73.152), (20.6833, 91.44)

// PID constants (adjust according to your needs)
#define KP 1.0
#define KI 0.0
#define KD 0.0
using namespace cv;
using namespace std;

mavros_msgs::State current_state;
geometry_msgs::PoseStamped current_pose;

void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}

void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_pose = *msg;
}

double compute_pid(double setpoint, double current, double& prev_error, double integral) {
    double error = setpoint - current;
    integral += error;
    double derivative = error - prev_error;
    double output = KP * error + KI * integral + KD * derivative;
    prev_error = error;
    return output;
}
class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Subscriber depth_image_sub_;

  image_transport::Publisher image_pub_;

  cv::Mat depth_mask;
  bool is_depth_mask = false;
  string img_name;

public:
  ImageConverter(string my_name)
    : it_(nh_),  img_name(my_name)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1,
      &ImageConverter::imageCb, this);
    // depth_image_sub_ = it_.subscribe("/camera/depth/image_raw", 1,
    //   &ImageConverter::depthImageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);

    cv::namedWindow("source");
  }

  ~ImageConverter()
  {
    cv::destroyWindow("source");
  }
  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    ROS_INFO("Image received");
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat image = cv_ptr->image;
    // cv::cvtColor(image, image, COLOR_BGR2RGB);
    // namedWindow("source", WINDOW_AUTOSIZE);
    cv::imshow("source", image);
    imwrite(img_name, image);
    cv::waitKey(3); 
    // image_pub_.publish(msg_out);
  }
};
// public:
//   ImageConverter()
//     : it_(nh_)
//   {
//     // Subscrive to input video feed and publish output video feed
//     image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1,
//       &ImageConverter::imageCb, this);
//     depth_image_sub_ = it_.subscribe("/camera/depth/image_raw", 1,
//       &ImageConverter::depthImageCb, this);
//     image_pub_ = it_.advertise("/image_converter/output_video", 1);

//     cv::namedWindow("source");
//   }

//   ~ImageConverter()
//   {
//     cv::destroyWindow("source");
//   }
// }


int main(int argc, char **argv) {
    ros::init(argc, argv, "takeoff_hover_land");
    ros::NodeHandle nh;

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>(
            "mavros/state", 10, state_cb);
    ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>(
            "mavros/local_position/pose", 10, pose_cb);
    ros::Publisher local_pos_pub = nh.advertise<mavros_msgs::PositionTarget>(
            "mavros/setpoint_raw/local", 10);
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>(
            "mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>(
            "mavros/set_mode");
    ros::ServiceClient land_client = nh.serviceClient<mavros_msgs::CommandTOL>(
            "mavros/cmd/land");
    // ros::Subscriber camera_sub = nh.subscribe<&ImageConverter::imageCb>("/camera/rgb/image_raw");
    // ImageConverter ic;

    // ros::Subscriber camera_sub = nh.subscribe<mavros_msgs::imageCb>("/camera/rgb/image_raw", 1, state_im);

    // Hey!! How are things lol
    // trying to get the camera to actually take a picture, 
    // we're still pushing under your github
    // how's the masters going
    //
    // Great! Looks like you guys have a lot 
    // of sweet work for suas. When's the competition? 
    //
    // this summer in maryland, I think its near the end of june
    ros::Rate rate(20.0);

    // Wait for FCU connection
    while (ros::ok() && current_state.connected) {
        ros::spinOnce();
        rate.sleep();
        ROS_INFO("Connecting to FCU...");
    }

    mavros_msgs::PositionTarget target;
    target.type_mask = target.IGNORE_AFX | target.IGNORE_AFY | target.IGNORE_AFZ | target.IGNORE_VX | target.IGNORE_VY | target.IGNORE_VZ | target.IGNORE_YAW_RATE;

    // Send a few setpoints before starting
    for (int i = 100; ros::ok() && i > 0; --i) {
        local_pos_pub.publish(target);
        ros::spinOnce();
        rate.sleep();
    }

    // Change to offboard mode and arm
    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();

    while (ros::ok() && !current_state.armed) {
        if (current_state.mode != "OFFBOARD" && (ros::Time::now() - last_request > ros::Duration(5.0))) {
            if (set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent) {
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        } else {
            if (!current_state.armed && (ros::Time::now() - last_request > ros::Duration(5.0))) {
                if (arming_client.call(arm_cmd) && arm_cmd.response.success) {
                    ROS_INFO("Vehicle armed");
                }
                last_request = ros::Time::now();
            }
        }
        local_pos_pub.publish(target);
        ros::spinOnce();
        rate.sleep();
    }

    // Start the image stuff
    // (You can add your image processing code here if needed)

    // Go to the first waypoint (hover)
    // ros::Time start_time = ros::Time::now();
    // double integral_z = 0.0;
    // double prev_error_z = 0.0;

    // double ImagePoints[5][3] = {{20.6833, 18.288, 38}, {20.6833, 36.576, 38}, {20.6833, 54.864, 38}, {20.6833, 73.152, 38}, {20.6833, 91.44, 38}};
    double ImagePoints[4][3] = {{20.6833, 21.9456, 38}, {20.6833, 43.8912, 38}, {20.6833, 65.8368, 38}, {20.6833, 87.7824, 38}};
    std::vector<geometry_msgs::PoseStamped> waypoints;
    for (const auto point : ImagePoints) {
            geometry_msgs::PoseStamped waypoint;

            waypoint.pose.position.x = point[0];
            waypoint.pose.position.y = point[1];
            waypoint.pose.position.z = point[2];

            // ROS_INFO("Reading waypoints from txt file: x=%f, y=%f, z=%f", lat, lon, alt_ft);
            waypoints.push_back(waypoint);
        }

    // std::vector<geometry_msgs::PoseStamped> waypoints = [];
    int counter = 1;
    for (const auto& waypoint : waypoints) {
        string temp = "test_" + to_string(counter) + "_.jpg";
        ImageConverter ic(temp);
        ros::Time waypoint_start_time = ros::Time::now();
        double lat = waypoint.pose.position.x;
        double lon = waypoint.pose.position.y;
        double alt_ft = waypoint.pose.position.z;
        ROS_INFO("Flying to the following waypoint: x=%f, y=%f, z=%f", lat, lon, alt_ft);

        while (ros::ok() && (ros::Time::now() - waypoint_start_time).toSec() < 20.0) {
            target.position.x = waypoint.pose.position.x;
            target.position.y = waypoint.pose.position.y;
            target.position.z = waypoint.pose.position.z;

            local_pos_pub.publish(target);
            ros::spinOnce();
            rate.sleep();
        }
        counter++;
    }
    // for(int i = 0; i < 5; ++i) {
    //   while (ros::ok() && (ros::Time::now() - start_time).toSec() < HOVER_DURATION) {
    //     // Compute PID for z-axis
    //     double target_z = FLIGHT_ALTITUDE;
    //     double current_z = current_pose.pose.position.z;
    //     double output_z = compute_pid(target_z, current_z, prev_error_z, integral_z);

    //     // Update position target
    //     target.position.x = ;
    //     target.position.y = HOVER_Y;
    //     target.position.z = FLIGHT_ALTITUDE + output_z;

    //     local_pos_pub.publish(target);
    //     ros::spinOnce();
    //     rate.sleep();
    //   }

    // }

    // // Go to the first waypoint (hover)
    // ros::Time start_time = ros::Time::now();
    // double integral_z = 0.0;
    // double prev_error_z = 0.0;

    // while (ros::ok() && (ros::Time::now() - start_time).toSec() < HOVER_DURATION) {
    //     // Compute PID for z-axis
    //     double target_z = FLIGHT_ALTITUDE;
    //     double current_z = current_pose.pose.position.z;
    //     double output_z = compute_pid(target_z, current_z, prev_error_z, integral_z);

    //     // Update position target
    //     target.position.x = HOVER_X;
    //     target.position.y = HOVER_Y;
    //     target.position.z = FLIGHT_ALTITUDE + output_z;

    //     local_pos_pub.publish(target);
    //     ros::spinOnce();
    //     rate.sleep();
    // }

    // Land
    mavros_msgs::CommandTOL land_cmd;
    land_cmd.request.yaw = 0;
    land_cmd.request.latitude = 0;
    land_cmd.request.longitude = 0;
    land_cmd.request.altitude = 0;

    if (land_client.call(land_cmd) && land_cmd.response.success) {
        ROS_INFO("Landing...");
    }

    return 0;
}
