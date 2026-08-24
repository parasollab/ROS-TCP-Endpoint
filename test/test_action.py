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

from types import SimpleNamespace
from unittest.mock import Mock, patch

from ros_tcp_endpoint.action import RosAction


class ImmediateFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        callback(self)


class DeferredFuture(ImmediateFuture):
    def __init__(self, value):
        super().__init__(value)
        self.callback = None

    def add_done_callback(self, callback):
        self.callback = callback

    def complete(self):
        self.callback(self)


class FakeSender:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        def record(*args):
            self.events.append((name, args))

        return record


class FakeServer:
    def __init__(self):
        self.unity_tcp_sender = FakeSender()
        self.errors = []

    def logerr(self, message):
        self.errors.append(message)


class FakeActionType:
    class Goal:
        pass


def make_goal_handle(accepted=True, result_future=None):
    if result_future is None:
        result_future = ImmediateFuture(
            SimpleNamespace(status=4, result="terminal-result")
        )
    handle = Mock()
    handle.accepted = accepted
    handle.goal_id = SimpleNamespace(uuid=bytes(range(16)))
    handle.get_result_async.return_value = result_future
    handle.cancel_goal_async.return_value = ImmediateFuture(
        SimpleNamespace(return_code=0)
    )
    return handle


def make_action(action_client):
    server = FakeServer()
    with patch("ros_tcp_endpoint.action.ActionClient", return_value=action_client):
        action = RosAction(
            "/execute_task_solution", FakeActionType, "pkg/Execute", server
        )
    return action, server


def event_names(server):
    return [event[0] for event in server.unity_tcp_sender.events]


def test_acceptance_feedback_and_terminal_result_are_relayed():
    action_client = Mock()
    action_client.server_is_ready.return_value = True
    handle = make_goal_handle()

    def send_goal(goal, feedback_callback):
        feedback_callback(SimpleNamespace(feedback="feedback"))
        return ImmediateFuture(handle)

    action_client.send_goal_async.side_effect = send_goal
    action, server = make_action(action_client)

    with patch("ros_tcp_endpoint.action.deserialize_message", return_value="goal"):
        action.send_goal("unity-guid", b"cdr")

    assert event_names(server) == [
        "send_action_goal_response",
        "send_action_feedback",
        "send_action_result",
    ]
    goal_response = server.unity_tcp_sender.events[0][1]
    assert goal_response[:3] == (
        "/execute_task_solution",
        "unity-guid",
        True,
    )
    assert "unity-guid" not in action.goal_handles


def test_rejected_goal_has_no_result_request():
    action_client = Mock()
    action_client.server_is_ready.return_value = True
    handle = make_goal_handle(accepted=False)
    action_client.send_goal_async.return_value = ImmediateFuture(handle)
    action, server = make_action(action_client)

    with patch("ros_tcp_endpoint.action.deserialize_message", return_value="goal"):
        action.send_goal("rejected", b"cdr")

    assert event_names(server) == ["send_action_goal_response"]
    assert server.unity_tcp_sender.events[0][1][2] is False
    handle.get_result_async.assert_not_called()


def test_unavailable_server_returns_explicit_error():
    action_client = Mock()
    action_client.server_is_ready.return_value = False
    action, server = make_action(action_client)

    action.send_goal("unavailable", b"cdr")

    assert event_names(server) == ["send_action_error"]
    assert server.unity_tcp_sender.events[0][1][2] == "server_unavailable"
    action_client.send_goal_async.assert_not_called()


def test_cancel_and_concurrent_goals_are_correlated():
    action_client = Mock()
    action_client.server_is_ready.return_value = True
    result_one = DeferredFuture(SimpleNamespace(status=5, result="one"))
    result_two = DeferredFuture(SimpleNamespace(status=4, result="two"))
    handles = [make_goal_handle(result_future=result_one),
               make_goal_handle(result_future=result_two)]
    action_client.send_goal_async.side_effect = [
        ImmediateFuture(handles[0]),
        ImmediateFuture(handles[1]),
    ]
    action, server = make_action(action_client)

    with patch("ros_tcp_endpoint.action.deserialize_message", return_value="goal"):
        action.send_goal("goal-one", b"one")
        action.send_goal("goal-two", b"two")
    action.cancel_goal("goal-one")

    assert set(action.goal_handles) == {"goal-one", "goal-two"}
    assert event_names(server).count("send_action_goal_response") == 2
    assert event_names(server).count("send_action_cancel_response") == 1

    result_two.complete()
    result_one.complete()
    result_events = [
        event for event in server.unity_tcp_sender.events
        if event[0] == "send_action_result"
    ]
    assert [event[1][1] for event in result_events] == ["goal-two", "goal-one"]
    assert action.goal_handles == {}


def test_unknown_cancel_returns_explicit_error():
    action_client = Mock()
    action, server = make_action(action_client)

    action.cancel_goal("missing")

    assert server.unity_tcp_sender.events[0][0] == "send_action_error"
    assert server.unity_tcp_sender.events[0][1][2] == "unknown_goal"
