# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/apps/cmake/2.8.9/bin/cmake

# The command to remove a file.
RM = /opt/apps/cmake/2.8.9/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /opt/apps/cmake/2.8.9/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default

# Include any dependencies generated for this target.
include apps/mypagerank/CMakeFiles/pagerank.out.dir/depend.make

# Include the progress variables for this target.
include apps/mypagerank/CMakeFiles/pagerank.out.dir/progress.make

# Include the compile flags for this target's objects.
include apps/mypagerank/CMakeFiles/pagerank.out.dir/flags.make

apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o: apps/mypagerank/CMakeFiles/pagerank.out.dir/flags.make
apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o: ../../apps/mypagerank/pagerank.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o"
	cd /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/apps/mypagerank && /opt/apps/gcc/4.7.1/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/pagerank.out.dir/pagerank.cpp.o -c /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/apps/mypagerank/pagerank.cpp

apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pagerank.out.dir/pagerank.cpp.i"
	cd /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/apps/mypagerank && /opt/apps/gcc/4.7.1/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/apps/mypagerank/pagerank.cpp > CMakeFiles/pagerank.out.dir/pagerank.cpp.i

apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pagerank.out.dir/pagerank.cpp.s"
	cd /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/apps/mypagerank && /opt/apps/gcc/4.7.1/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/apps/mypagerank/pagerank.cpp -o CMakeFiles/pagerank.out.dir/pagerank.cpp.s

apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o.requires:
.PHONY : apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o.requires

apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o.provides: apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o.requires
	$(MAKE) -f apps/mypagerank/CMakeFiles/pagerank.out.dir/build.make apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o.provides.build
.PHONY : apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o.provides

apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o.provides.build: apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o

apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o: apps/mypagerank/CMakeFiles/pagerank.out.dir/flags.make
apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o: ../../apps/mypagerank/pr_spMatrix.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o"
	cd /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/apps/mypagerank && /opt/apps/gcc/4.7.1/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o -c /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/apps/mypagerank/pr_spMatrix.cpp

apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.i"
	cd /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/apps/mypagerank && /opt/apps/gcc/4.7.1/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/apps/mypagerank/pr_spMatrix.cpp > CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.i

apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.s"
	cd /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/apps/mypagerank && /opt/apps/gcc/4.7.1/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/apps/mypagerank/pr_spMatrix.cpp -o CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.s

apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o.requires:
.PHONY : apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o.requires

apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o.provides: apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o.requires
	$(MAKE) -f apps/mypagerank/CMakeFiles/pagerank.out.dir/build.make apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o.provides.build
.PHONY : apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o.provides

apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o.provides.build: apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o

# Object files for target pagerank.out
pagerank_out_OBJECTS = \
"CMakeFiles/pagerank.out.dir/pagerank.cpp.o" \
"CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o"

# External object files for target pagerank.out
pagerank_out_EXTERNAL_OBJECTS =

apps/mypagerank/pagerank.out: apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o
apps/mypagerank/pagerank.out: apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o
apps/mypagerank/pagerank.out: apps/mypagerank/CMakeFiles/pagerank.out.dir/build.make
apps/mypagerank/pagerank.out: src/libgalois.a
apps/mypagerank/pagerank.out: /usr/lib64/libnuma.so
apps/mypagerank/pagerank.out: apps/mypagerank/CMakeFiles/pagerank.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable pagerank.out"
	cd /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/apps/mypagerank && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pagerank.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
apps/mypagerank/CMakeFiles/pagerank.out.dir/build: apps/mypagerank/pagerank.out
.PHONY : apps/mypagerank/CMakeFiles/pagerank.out.dir/build

apps/mypagerank/CMakeFiles/pagerank.out.dir/requires: apps/mypagerank/CMakeFiles/pagerank.out.dir/pagerank.cpp.o.requires
apps/mypagerank/CMakeFiles/pagerank.out.dir/requires: apps/mypagerank/CMakeFiles/pagerank.out.dir/pr_spMatrix.cpp.o.requires
.PHONY : apps/mypagerank/CMakeFiles/pagerank.out.dir/requires

apps/mypagerank/CMakeFiles/pagerank.out.dir/clean:
	cd /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/apps/mypagerank && $(CMAKE_COMMAND) -P CMakeFiles/pagerank.out.dir/cmake_clean.cmake
.PHONY : apps/mypagerank/CMakeFiles/pagerank.out.dir/clean

apps/mypagerank/CMakeFiles/pagerank.out.dir/depend:
	cd /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1 /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/apps/mypagerank /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/apps/mypagerank /home1/03153/ani91/hw7/galois_pagerank/Galois-2.2.1/build/default/apps/mypagerank/CMakeFiles/pagerank.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : apps/mypagerank/CMakeFiles/pagerank.out.dir/depend
