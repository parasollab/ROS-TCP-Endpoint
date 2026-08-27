#!/usr/bin/env python3
"""
Known-good reference for the CollisionObject path, so "is Unity's message structurally correct?"
becomes a diff instead of a judgement call.

It builds a moveit_msgs/CollisionObject with the same shape PlanningSceneBridge builds - identity
object pose, one shape_msgs/Mesh whose vertices are already in the planning frame, one identity
mesh pose, operation ADD - but with a four-vertex tetrahedron instead of a room, so the serialized
bytes are short enough to read by eye.

    # what correct bytes look like, and prove the ROS side works with no headset involved
    python3 tools/collision_object_probe.py --publish

    # same message through the service path PlanningSceneBridge uses in ApplyService mode
    python3 tools/collision_object_probe.py --apply

    # print anything that lands on /collision_object, including what Unity publishes
    python3 tools/collision_object_probe.py --listen

Compare the --publish hexdump against what the endpoint logs for Unity's message
(ROS_TCP_DEBUG_MSGS=1). Both should begin 00 01 00 00, the CDR encapsulation header. If Unity's
does not, its ROS2 scripting define is off and nothing downstream will be trustworthy.
"""

import argparse
import importlib.util
import os

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message

from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import Header

# Loaded by path rather than as ros_tcp_endpoint.msg_debug so this tool runs straight from a source
# checkout, without the package having to be built or installed first.
_spec = importlib.util.spec_from_file_location(
    "msg_debug", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "ros_tcp_endpoint", "msg_debug.py")
)
msg_debug = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msg_debug)

TOPIC = "/collision_object"
APPLY_SERVICE = "/apply_planning_scene"


def identity_pose():
    pose = Pose()
    # w must be 1.0. An all-zero quaternion is not a rotation, and MoveIt either warns and
    # normalises or rejects the object outright - a common silent failure when poses are built
    # field by field rather than from an identity constructor.
    pose.orientation.w = 1.0
    return pose


def reference_object(frame, object_id):
    """A tetrahedron: 4 vertices, 4 triangles. Small enough that every byte is inspectable."""
    mesh = Mesh()
    mesh.vertices = [
        Point(x=0.0, y=0.0, z=0.0),
        Point(x=0.1, y=0.0, z=0.0),
        Point(x=0.0, y=0.1, z=0.0),
        Point(x=0.0, y=0.0, z=0.1),
    ]
    mesh.triangles = [
        MeshTriangle(vertex_indices=[0, 2, 1]),
        MeshTriangle(vertex_indices=[0, 1, 3]),
        MeshTriangle(vertex_indices=[0, 3, 2]),
        MeshTriangle(vertex_indices=[1, 2, 3]),
    ]

    obj = CollisionObject()
    obj.header = Header(frame_id=frame)
    obj.id = object_id
    obj.operation = CollisionObject.ADD
    obj.pose = identity_pose()
    obj.meshes = [mesh]
    obj.mesh_poses = [identity_pose()]
    return obj


def show(label, obj):
    raw = serialize_message(obj)
    print("\n=== {} ===".format(label))
    print("{} bytes, {}".format(len(raw), msg_debug.cdr_verdict(raw)))
    print("raw: {}".format(msg_debug.hexdump(raw, limit=96)))
    print(msg_debug.describe(obj, max_seq=4))
    return raw


class Publisher(Node):
    def __init__(self, args):
        super().__init__("collision_object_probe_publisher")
        obj = reference_object(args.frame, args.id)
        show("reference CollisionObject as this node built it", obj)

        pub = self.create_publisher(CollisionObject, TOPIC, 10)
        # PlanningSceneMonitor subscribes lazily; publishing into a topic with no subscriber yet
        # drops the message and looks exactly like a rejected object.
        self.create_timer(0.5, lambda: pub.publish(obj))
        self.get_logger().info(
            "Publishing '{}' on {} every 0.5 s. It should appear in RViz's Planning Scene "
            "display and in: ros2 service call /get_planning_scene "
            "moveit_msgs/srv/GetPlanningScene \"{{components: {{components: 8}}}}\"".format(
                args.id, TOPIC
            )
        )


class Applier(Node):
    def __init__(self, args):
        super().__init__("collision_object_probe_applier")
        obj = reference_object(args.frame, args.id)
        show("reference CollisionObject as this node built it", obj)

        client = self.create_client(ApplyPlanningScene, APPLY_SERVICE)
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("{} never appeared - is move_group running?".format(APPLY_SERVICE))
            raise SystemExit(1)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]

        request = ApplyPlanningScene.Request()
        request.scene = scene

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        if not future.done():
            self.get_logger().error("{} did not answer within 20 s.".format(APPLY_SERVICE))
            raise SystemExit(1)
        # Acknowledged is the whole point of the service path: unlike the topic, a false here is a
        # real rejection rather than a message nobody happened to be listening for.
        self.get_logger().info("{} returned success={}".format(APPLY_SERVICE, future.result().success))


class Listener(Node):
    def __init__(self, args):
        super().__init__("collision_object_probe_listener")
        self.create_subscription(CollisionObject, TOPIC, self.on_object, 50)
        self.get_logger().info("Listening on {}. Publish from Unity now.".format(TOPIC))

    def on_object(self, obj):
        print("\n=== received on {} ===".format(TOPIC))
        print(msg_debug.describe(obj, max_seq=4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true", help="publish a reference object on " + TOPIC)
    mode.add_argument("--apply", action="store_true", help="send it through " + APPLY_SERVICE)
    mode.add_argument("--listen", action="store_true", help="print objects arriving on " + TOPIC)
    parser.add_argument("--frame", default="panda_link0", help="planning frame (default: panda_link0)")
    parser.add_argument("--id", default="probe_reference_tetra", help="collision object id")
    args = parser.parse_args()

    rclpy.init()
    try:
        if args.publish:
            node = Publisher(args)
        elif args.apply:
            Applier(args)
            return
        else:
            node = Listener(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
