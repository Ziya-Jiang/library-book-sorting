# cuarm_robotics
## Tested System
  * Linux(Amd64): Ubuntu18.04, Ubuntu20.04, Ubuntu22.04
## Install dependencies
CMake is used to support cross-platform building. Install [CMake](http://www.cmake.org) (Resources -> Download) before proceeding.
* ## yaml-cpp (https://github.com/jbeder/yaml-cpp)
  * ### On Linux
    Follow instructions on yaml-cpp github.
    ```bash
    mkdir build 
    cd build
    make
    make install
    ```
  * ### On window
   1. **Get yaml-cpp from github**
   3. **Build with CMake**
      ```bash
      mkdir build
      cd build
      cmake .. -G "Visual Studio 16 2019"   #(VS 2019 64-bits)
      ```
   3. **Generate library using Visual Studio** (https://ccf19881030.blog.csdn.net/article/details/108676484)
     * Navigate into the build directory, open YAML_CPP.sln with Visual Studio.
     * Select Release/X64, then press Build ALL_BUILD(Ctrl+B).
     * yaml-cpp.lib is generated under build/Release directory.
   4. **Configure environment** (Create the environment variable if it does not exist)
     * Add "*<your_yaml-cpp_path>\include;*" to user environment variable *INCLUDE_PATH*.
     * Add "*<your_yaml-cpp_path>\build\Release;*" to user environment variable *LIBRARY_PATH*.

* ## Pinocchio
  * ### On Linux
    Follow instructions on [Pinocchio](https://stack-of-tasks.github.io/pinocchio/download.html) website.
  * ### On Window
   1. **Install [MiniConda](https://docs.conda.io/en/latest/miniconda.html)** 
   2. **Install Pinocchio in Conda**
      ```bash
      conda install -c conda-forge pinocchio
      ```
   3. **Configure environment**  
     Add "*C:\Users\\\<username>\miniconda3\Library\bin;*" to user environment variable *Path*

## Built it
```bash
git clone https://github.com/robot-chenwei/differential_module.git
```
* ### On Linux  
  ```bash
  mkdir build
  cd build
  cmake ..
  make
  ```
* ### On Window  
  Compile with *Visual Studio Build Tool* in vscode.(https://blog.csdn.net/weixin_43669941/article/details/108921714)
  * Confirm vscode is in *Release* build. 
  * Confirm vscode is build with *Visual Studio Build Tool <version> Release - amd64*. 
  * F7 (CMake: Build)
  * Ctrl + F5 (Run)

## Run the demo
  * ### On Linux  
  ```bash
  ./cuarm_robotics
  ```
  * ### On Window  
  Navigate into the build\Release directory, double click *differential_module.exe*.

<!-- ![panel](https://user-images.githubusercontent.com/34245169/149092762-ef7f6ded-50a6-4d2e-8cae-6ff794332fea.png) -->
