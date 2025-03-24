cd fasterlio
cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release && cmake --build build
cd ../process_module
cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release && cmake --build build
cd ../Traj-LO
cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release && cmake --build build