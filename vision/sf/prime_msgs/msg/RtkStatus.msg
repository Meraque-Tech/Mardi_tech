std_msgs/Header header
bool rtk_fix # currently used for debugging 
string carrier_soln # rtk no fix, float, fixed 
int64 num_sats # number of satellites used in solution
int64 num_sats_above_40db # number of satellites above 40db just to indicate the signals strength
int64 rtcm_msg_type # rtcm message type
bool rtcm_is_receiving # is recieving rtcm messages

# string num_sats # number of satellites used in navigation solution
# string sats_above_40db # list of satellites above 40db
# string sats_above_30db # list of satellites above 30db
# string sats_below_30db # list of satellites below 30db
