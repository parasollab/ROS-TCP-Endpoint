#!/usr/bin/env python3
# Copyright 2026 Unity Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Small ROS 2 action server used by the Unity action-client sample."""

import time

import rclpy
from example_interfaces.action import Fibonacci
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class FibonacciActionServer(Node):
    """Exercise success, rejection, feedback, cancellation, and abort paths."""

    ABORT_ORDER = 13

    def __init__(self):
        super().__init__("fibonacci_action_server")
        self.declare_parameter("feedback_period", 0.3)
        self._action_server = ActionServer(
            self,
            Fibonacci,
            "fibonacci",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            "Ready on /fibonacci. Orders <= 0 are rejected; order 13 aborts."
        )

    def goal_callback(self, goal_request):
        if goal_request.order <= 0:
            self.get_logger().info(
                "Rejecting Fibonacci goal with order %d" % goal_request.order
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            "Accepting Fibonacci goal with order %d" % goal_request.order
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        self.get_logger().info("Accepting cancellation request")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        order = goal_handle.request.order
        sequence = [0] if order == 1 else [0, 1]
        period = self.get_parameter("feedback_period").value

        while len(sequence) < order:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("Goal canceled")
                return self._result(sequence)

            sequence.append(sequence[-1] + sequence[-2])
            feedback = self._feedback(sequence)
            goal_handle.publish_feedback(feedback)
            self.get_logger().info("Feedback: %s" % sequence)

            if order == self.ABORT_ORDER and len(sequence) >= 5:
                goal_handle.abort()
                self.get_logger().info("Aborting order-13 demo goal")
                return self._result(sequence)

            time.sleep(period)

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self.get_logger().info("Goal canceled")
        else:
            goal_handle.succeed()
            self.get_logger().info("Goal succeeded: %s" % sequence)
        return self._result(sequence)

    @staticmethod
    def _feedback(sequence):
        feedback = Fibonacci.Feedback()
        feedback.sequence = sequence.copy()
        return feedback

    @staticmethod
    def _result(sequence):
        result = Fibonacci.Result()
        result.sequence = sequence
        return result

    def destroy_node(self):
        self._action_server.destroy()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FibonacciActionServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
