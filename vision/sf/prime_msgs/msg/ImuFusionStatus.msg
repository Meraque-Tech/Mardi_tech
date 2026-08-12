# Temporary msg to validate and debug the imu heading fusion method
std_msgs/Header header
bool valid_correction # Flag to indicate if the imu is getting a valid correction
float64 heading_correction # Heading correction to be applied to the imu heading