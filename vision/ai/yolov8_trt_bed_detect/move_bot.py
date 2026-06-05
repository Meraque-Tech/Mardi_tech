#! /usr/bin/env python3
import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.clock import Clock
from std_msgs.msg import UInt8
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from time import sleep
import os 
import sys
from lytbot2_msg_srv.srv import FlashSize, Start, FlashPointReached, NextFlashPoint

class Goal_pose_nav(Node):
    def __init__(self):
        super().__init__('Goal_pose_nav')

        self.font_pose_array_sub = self.create_subscription(Path, 'font_pose_array', self.font_pose_array_cb, 10)
        self.left_pose_array_sub = self.create_subscription(Path, 'left_pose_array', self.left_pose_array_cb, 10)
        self.right_pose_array_sub = self.create_subscription(Path, 'right_pose_array', self.right_pose_array_cb, 10)
        # self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._action_client = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')
        
        

        self.flash_size_srv = self.create_service(
            FlashSize,
            'flash_size',
            self.flash_size_callback_srv
        )

        self.start_srv = self.create_service(
            Start,
            'start',
            self.start_cb_srv
        )

        self.next_flashpoint_srv = self.create_service(
            NextFlashPoint,
            'next_flashpoint',
            self.next_flashpoint_callback_srv)

        self.flashpoint_reached_client = self.create_client(FlashPointReached, 'reached_fp')


        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.ic = 0

        self.goal_cnt = 0
        self.font_poses = []
        self.left_poses = []
        self.right_poses = []

        self.font_poses_flag = 0
        self.left_poses_flag = 0
        self.right_poses_flag = 0

        self.nop = 3
        self.flash_done_ = 0
        
    
    def flash_size_callback_srv(self, request, response):
        if request.flash_size == 1:
            self.goal_cnt = len(self.lrf_poses)
            print("flash size ----> ", self.goal_cnt)
            response.flash_size_fb = self.goal_cnt
            return response
            
    def start_cb_srv(self, request, response):
        if request.start == 1:
            print("<-------------------------publish_path---------------------------->", self.ic)
            if(self.ic < self.goal_cnt):
                self.send_multiple_goal(self.lrf_poses[self.ic])
                response.start_fb = 1
            return response
        
    def next_flashpoint_callback_srv(self, request, response):
        if request.next_flash_point == 1:
            print("<-------------------------publish_path---------------------------->", self.ic)
            if(self.ic < self.goal_cnt):
                self.send_multiple_goal(self.lrf_poses[self.ic])
                response.next_flash_point_fb = 1
            return response


    def paith_gen(self, l_alpha, r_alpha, f_alpha):
        
        if l_alpha == 0 and r_alpha == 0 and f_alpha == 1:
            move_poses_c1 = [self.font_poses]
            self.lrf_poses = [move_poses_c1]
            self.ic = 0

        
        if l_alpha == 0 and r_alpha == 1 and f_alpha == 0:
            move_poses_c1 = self.path_generator(self.right_poses[0], self.right_poses[1], self.nop)
            self.lrf_poses = [move_poses_c1]
            self.ic = 0

        if l_alpha == 0 and r_alpha == 1 and f_alpha == 1:
            move_poses_c1 = self.path_generator(self.right_poses[0], self.right_poses[1], self.nop)
            move_poses_c2 = self.path_generator(self.right_poses[1], self.right_poses[2], self.nop) + self.font_poses
            self.lrf_poses = [move_poses_c1, move_poses_c2]
            self.ic = 0

        if l_alpha == 1 and r_alpha == 0 and f_alpha == 0:
            move_poses_c1 = self.path_generator(self.left_poses[0], self.left_poses[1], self.nop)
            self.lrf_poses = [move_poses_c1]
            self.ic = 0
            
        if l_alpha == 1 and r_alpha == 0 and f_alpha == 1:
            move_poses_c1 = self.path_generator(self.left_poses[0], self.left_poses[1], self.nop)
            move_poses_c2 = self.path_generator(self.left_poses[1], self.left_poses[2], self.nop) + self.font_poses
            self.lrf_poses = [move_poses_c1, move_poses_c2]
            self.ic = 0

        if l_alpha == 1 and r_alpha == 1 and f_alpha == 0:
            move_poses_c1 = self.path_generator(self.left_poses[0], self.left_poses[1], self.nop)
            move_poses_c2_part1 = self.path_generator(self.left_poses[1], self.left_poses[2], self.nop)
            move_poses_c2_part2 = self.path_generator(self.left_poses[2], self.right_poses[0], self.nop)
            move_poses_c2_part3 = self.path_generator(self.right_poses[0], self.right_poses[1], self.nop)
            move_poses_c2 = move_poses_c2_part1 + move_poses_c2_part2 + move_poses_c2_part3
            self.lrf_poses = [move_poses_c1, move_poses_c2]
            self.ic = 0

        if l_alpha == 1 and r_alpha == 1 and f_alpha == 1:
            move_poses_c1 = self.path_generator(self.left_poses[0], self.left_poses[1], self.nop)
            move_poses_c2_part1 = self.path_generator(self.left_poses[1], self.left_poses[2], self.nop)
            move_poses_c2_part2 = self.path_generator(self.left_poses[2], self.right_poses[0], self.nop)
            move_poses_c2_part3 = self.path_generator(self.right_poses[0], self.right_poses[1], self.nop)
            move_poses_c2 = move_poses_c2_part1 + move_poses_c2_part2 + move_poses_c2_part3
            self.lrf_poses = [move_poses_c1, move_poses_c2]
            self.ic = 0

    
    def timer_callback(self):
        self.paith_gen(self.left_poses_flag, self.right_poses_flag, self.font_poses_flag)

        self.font_poses_flag = 0
        self.left_poses_flag = 0
        self.right_poses_flag = 0

        self.font_poses = []
        self.left_poses = []
        self.right_poses = []
    
    def path_generator(self, move_poses_i, move_poses_f, NOP):
        xi = move_poses_i.pose.position.x
        yi = move_poses_i.pose.position.y
        xi_oz = move_poses_i.pose.orientation.z
        xi_ow = move_poses_i.pose.orientation.w

        xf = move_poses_f.pose.position.x
        yf = move_poses_f.pose.position.y
        xf_oz = move_poses_f.pose.orientation.z
        xf_ow = move_poses_f.pose.orientation.w

        move_poses = [move_poses_i]
        for j in range(0, NOP):
            PX = (j * xf + (NOP - j) * xi) / NOP
            PY = (j * yf + (NOP - j) * yi) / NOP
            move_poses.append(self.create_pose_stamped(PX, PY, 0.0, 0.0, 0.0, xf_oz, xf_ow, move_poses_f))
        move_poses.append(move_poses_f)
        return move_poses

    def create_pose_stamped(self, x, y, z, qx, qy, qz, qw, poses_):
        pose_stamped = PoseStamped()
        pose_stamped.header = poses_.header

        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw

        return pose_stamped

    def font_pose_array_cb(self, msg):
        print("font len of the bed_pose array --> ", len(msg.poses))
        self.font_poses_flag = 1
        self.font_poses.append(msg.poses[0])

        # self.send_multiple_goal([self.goal_poses[self.ic], self.goal_poses[self.ic+1]])
        # self.send_goal(self.goal_poses[self.ic])
    
    def left_pose_array_cb(self, msg):
        print("left len of the bed_pose array --> ", len(msg.poses))
        self.left_poses_flag = 1
        self.left_poses.append(msg.poses[0])
        self.left_poses.append(msg.poses[1])
        self.left_poses.append(msg.poses[2])

    def right_pose_array_cb(self, msg):
        print("right len of the bed_pose array --> ", len(msg.poses))
        self.right_poses_flag = 1
        self.right_poses.append(msg.poses[0])
        self.right_poses.append(msg.poses[1])
        self.right_poses.append(msg.poses[2])
    
    def send_multiple_goal(self, pointStamps):
        goal_poses = NavigateThroughPoses.Goal()
        goal_poses.poses = pointStamps

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Action server detected.')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_poses,
            feedback_callback=self.feedback_callback)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
    
    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation succeeded! '+str(self.ic))
            self.ic +=1

            self.req = FlashPointReached.Request()
            self.req.flashpoint_id = self.ic  # Replace 'count' with your desired value
            self.flashpoint_reached_client.call_async(self.req)

        elif status == GoalStatus.STATUS_ABORTED:
            print("goal is Aborted")
            # self.ic +=1
        else:
            self.get_logger().info(
                'Navigation failed with status: {0}'.format(status))




def main(args=None):
    rclpy.init(args=args)
    goal_pose_nav = Goal_pose_nav()
    rclpy.spin(goal_pose_nav)
    goal_pose_nav.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()