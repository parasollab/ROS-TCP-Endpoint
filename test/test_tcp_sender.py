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

import json
from pathlib import Path
from queue import Queue
import struct
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


try:
    import rclpy  # noqa: F401
except ModuleNotFoundError:
    rclpy = ModuleType("rclpy")
    rclpy_node = ModuleType("rclpy.node")
    rclpy_serialization = ModuleType("rclpy.serialization")
    rclpy_node.Node = object
    rclpy_serialization.deserialize_message = lambda *_args: None
    rclpy_serialization.serialize_message = lambda *_args: b""
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = rclpy_node
    sys.modules["rclpy.serialization"] = rclpy_serialization

package = ModuleType("ros_tcp_endpoint")
package.__path__ = [str(Path(__file__).parents[1] / "ros_tcp_endpoint")]
sys.modules["ros_tcp_endpoint"] = package

from ros_tcp_endpoint.tcp_sender import UnityTcpSender  # noqa: E402
from ros_tcp_endpoint.client import ClientThread  # noqa: E402


class ActionErrorSenderTests(unittest.TestCase):
    def test_action_error_is_a_header_only_atomic_command(self):
        sender = UnityTcpSender(None)
        sender.queue = Queue()

        sender.send_action_error(
            "/fibonacci", "unity-goal", "server_unavailable", "Server is down"
        )

        frame = sender.queue.get_nowait()
        command_length = struct.unpack_from("<I", frame, 0)[0]
        command_end = 4 + command_length
        command = frame[4:command_end].decode("utf-8")
        json_length = struct.unpack_from("<I", frame, command_end)[0]
        json_start = command_end + 4
        params = json.loads(frame[json_start:json_start + json_length])

        self.assertEqual("__action_error", command)
        self.assertEqual("/fibonacci", params["action_name"])
        self.assertEqual("unity-goal", params["goal_id"])
        self.assertEqual("server_unavailable", params["code"])
        self.assertEqual("Server is down", params["message"])
        self.assertEqual(json_start + json_length, len(frame))

    def test_action_feedback_keeps_payload_in_same_queue_item(self):
        sender = UnityTcpSender(None)
        sender.queue = Queue()
        feedback = object()

        with patch.object(
            ClientThread, "serialize_message", return_value=b"feedback-cdr"
        ) as serialize:
            sender.send_action_feedback("/fibonacci", "unity-goal", feedback)

        serialize.assert_called_once_with("/fibonacci", feedback)
        self.assertTrue(sender.queue.get_nowait().endswith(b"feedback-cdr"))


if __name__ == "__main__":
    unittest.main()
