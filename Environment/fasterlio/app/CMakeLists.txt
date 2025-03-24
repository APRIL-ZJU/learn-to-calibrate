# add_executable(run_mapping_online run_mapping_online.cc)
# target_link_libraries(run_mapping_online
#         ${PROJECT_NAME} gflags
#         )
# add_executable(run_mapping_offline run_mapping_offline.cc)
# target_link_libraries(run_mapping_offline
        # ${PROJECT_NAME} gflags
        # )
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(L2CE bind.cpp)
target_link_libraries(L2CE PRIVATE
          faster_lio OpenMP::OpenMP_CXX
          )
# add_executable(main main.cpp)
# target_include_directories(main PRIVATE
        #   ${pybind11_INCLUDE_DIRS}
        #   )
# target_link_libraries(main PRIVATE
        #   faster_lio gflags ${pybind11_LIBRARIES} OpenMP::OpenMP_CXX
        #   )