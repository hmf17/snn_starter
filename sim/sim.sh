#!/bin/zsh

ros_setup=/opt/ros/kinetic/setup.zsh
esim_setup=~/sim_ws/devel/setup.zsh
temp_path=/tmp/out.bag
frame_path=$1

source $ros_setup
source $esim_setup

# Pre treatment
echo "->Start Pre Treatment"
gnome-terminal -x zsh -c "source $ros_setup; source $esim_setup; roscd esim_ros; python scripts/generate_stamps_file.py -i $frame_path -r $2"
echo "<-End Pre Treatment"

# Start roscore
echo "->Check Roscore"
ps_out=`ps -ef | grep roscore | grep -v 'grep'`
if [[ "$ps_out" != "" ]];then
	echo "<-Roscore Already Running"
else
	gnome-terminal -x sh -c "roscore; exec bash"
	echo "<-Start Roscore"
fi

# Waiting roscore to start
sleep 2s

# Start simulation
echo "->Start Simulation"
gnome-terminal -x zsh -c "rosrun esim_ros esim_node \
--data_source=2 \
--path_to_output_bag=$temp_path \
--path_to_data_folder=$frame_path \
--ros_publisher_frame_rate=60 \
--exposure_time_ms=10.0 \
--use_log_image=1 \
--log_eps=0.1 \
--contrast_threshold_pos=0.3 \
--contrast_threshold_neg=0.3"
echo "Succeed! Roscore can be shut down after simulation exits."
