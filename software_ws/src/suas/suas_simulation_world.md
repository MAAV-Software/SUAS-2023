To run  (all done within sudo)

terminal 1:
1. cd /mission9-2022/scripts
2. source start_gazebo.bash
3. roslaunch px4 posix_sitl.launch world:=/SUAS-2023/software_ws/src/suas/waypoints.world vehicle:=/iris_downward_depth_camera

terminal 2:
1. roscore

terminal 3: 
1. roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14540"

terminal 4: 
1. cd /SUAS-2023/software_ws
2. catkin_make
3. source ./devel/setup.bash
4. rosrun [workspace name] [desired file (check tab for options or look to software_ws/src/suas/CMakeLists for desired file)]

To view camera:
two terminals
1. rosrun rviz rviz
2. rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 map [my_frame] 100

then in rviz go into add

