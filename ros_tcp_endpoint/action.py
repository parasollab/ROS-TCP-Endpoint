#  Copyright 2020 Unity Technologies
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import threading

from rclpy.action import ActionClient
from rclpy.serialization import deserialize_message


class RosAction:
    """ROS 2 action client owned by the TCP endpoint."""

    def __init__(self, action_name, action_class, message_name, tcp_server):
        self.action_name = action_name
        self.action_class = action_class
        self.message_name = message_name
        self.tcp_server = tcp_server
        self.client = ActionClient(tcp_server, action_class, action_name)
        self.goal_handles = {}
        self.feedback_backlog = {}
        self.goal_lock = threading.Lock()

    def send_goal(self, goal_id, data):
        """Deserialize and asynchronously submit a goal received from Unity."""
        with self.goal_lock:
            if goal_id in self.goal_handles:
                self._send_error(goal_id, "duplicate_goal", "Goal id is already in use")
                return
            self.goal_handles[goal_id] = None
            self.feedback_backlog[goal_id] = []

        if not self.client.server_is_ready():
            self._remove_goal(goal_id)
            self._send_error(
                goal_id, "server_unavailable", "Action server is not ready"
            )
            return

        try:
            goal = deserialize_message(data, self.action_class.Goal)
            future = self.client.send_goal_async(
                goal, feedback_callback=lambda message: self._feedback(goal_id, message)
            )
            future.add_done_callback(
                lambda result: self._goal_response(goal_id, result)
            )
        except Exception as exc:
            self._remove_goal(goal_id)
            self._send_error(goal_id, "send_goal_failed", str(exc))

    def cancel_goal(self, goal_id):
        """Asynchronously request cancellation of an accepted Unity goal."""
        with self.goal_lock:
            exists = goal_id in self.goal_handles
            goal_handle = self.goal_handles.get(goal_id)

        if not exists:
            self._send_error(goal_id, "unknown_goal", "Goal id is not registered")
            return
        if goal_handle is None:
            self._send_error(
                goal_id, "goal_pending", "Goal acceptance is still pending"
            )
            return

        try:
            future = goal_handle.cancel_goal_async()
            future.add_done_callback(
                lambda result: self._cancel_response(goal_id, result)
            )
        except Exception as exc:
            self._send_error(goal_id, "cancel_failed", str(exc))

    def _goal_response(self, goal_id, future):
        try:
            goal_handle = future.result()
            if goal_handle is None or not goal_handle.accepted:
                self._remove_goal(goal_id)
                self.tcp_server.unity_tcp_sender.send_action_goal_response(
                    self.action_name, goal_id, False, ""
                )
                return

            with self.goal_lock:
                if goal_id not in self.goal_handles:
                    return
                self.goal_handles[goal_id] = goal_handle
                buffered_feedback = self.feedback_backlog.pop(goal_id, [])
                ros_goal_id = bytes(goal_handle.goal_id.uuid).hex()
                # Queue acceptance before any feedback for this goal, even if a very
                # fast action server published feedback before the future callback ran.
                self.tcp_server.unity_tcp_sender.send_action_goal_response(
                    self.action_name, goal_id, True, ros_goal_id
                )
                for feedback in buffered_feedback:
                    self.tcp_server.unity_tcp_sender.send_action_feedback(
                        self.action_name, goal_id, feedback
                    )
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(
                lambda result: self._result(goal_id, result)
            )
        except Exception as exc:
            self._remove_goal(goal_id)
            self._send_error(goal_id, "goal_response_failed", str(exc))

    def _feedback(self, goal_id, feedback_message):
        with self.goal_lock:
            if goal_id not in self.goal_handles:
                return
            if self.goal_handles[goal_id] is None:
                self.feedback_backlog[goal_id].append(feedback_message.feedback)
                return
        try:
            self.tcp_server.unity_tcp_sender.send_action_feedback(
                self.action_name, goal_id, feedback_message.feedback
            )
        except Exception as exc:
            self._send_error(goal_id, "feedback_failed", str(exc))

    def _result(self, goal_id, future):
        try:
            wrapped_result = future.result()
            self.tcp_server.unity_tcp_sender.send_action_result(
                self.action_name, goal_id, wrapped_result.status, wrapped_result.result
            )
        except Exception as exc:
            self._send_error(goal_id, "result_failed", str(exc))
        finally:
            self._remove_goal(goal_id)

    def _cancel_response(self, goal_id, future):
        try:
            response = future.result()
            self.tcp_server.unity_tcp_sender.send_action_cancel_response(
                self.action_name, goal_id, response.return_code
            )
        except Exception as exc:
            self._send_error(goal_id, "cancel_response_failed", str(exc))

    def _send_error(self, goal_id, code, message):
        self.tcp_server.unity_tcp_sender.send_action_error(
            self.action_name, goal_id, code, message
        )
        self.tcp_server.logerr(
            "Action '{}' goal '{}': {} ({})".format(
                self.action_name, goal_id, message, code
            )
        )

    def _remove_goal(self, goal_id):
        with self.goal_lock:
            self.goal_handles.pop(goal_id, None)
            self.feedback_backlog.pop(goal_id, None)

    def unregister(self):
        with self.goal_lock:
            self.goal_handles.clear()
            self.feedback_backlog.clear()
        self.client.destroy()
