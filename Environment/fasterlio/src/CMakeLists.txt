find_package(roscpp REQUIRED)
find_package(rosbag REQUIRED)
find_package(tf REQUIRED)
add_library(${PROJECT_NAME} SHARED
        laser_mapping.cc
        pointcloud_preprocess.cc
        options.cc
        utils.cc
        )
target_include_directories(${PROJECT_NAME} PUBLIC
        ../include
        ${rosbag_INCLUDE_DIRS}
 	${tf_INCLUDE_DIRS}
	${roscpp_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
)
# add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencpp livox_ros_driver_gencpp)

target_link_libraries(${PROJECT_NAME} PUBLIC
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${rosbag_LIBRARIES}
        ${tf_LIBRARIES}
        tbb
        glog
        yaml-cpp
        )
