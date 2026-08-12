# This message used for Dual Rtk values (TODO: Think of a better way to do this)

std_msgs/Header header
float64 heading # following ROS ENU convention, East is 0, and it increases counter-clockwise, hence north is 90 degrees 
float64 relative_dist # the distance between the two RTKs in meters (measured)
float64 baseline # the baseline between the two RTKs in meters (Real)

# the following are to measure the accuracy of RTK Dual
# the the system considered valid if both rtk are fixed
# and realtive distance lies between min_dist and max_dist, the mid point is the baseline

bool valid # whether both RTKs are fixed and the relative distance is valid
float64 min_valid_distance # whether the relative distance is less than the minimum distance
float64 max_valid_distance # whether the relative distance is greater than the maximum distance

