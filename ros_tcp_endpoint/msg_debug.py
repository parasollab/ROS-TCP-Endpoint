#  Diagnostics for messages arriving from Unity.
#
#  The endpoint deserializes whatever bytes Unity sends straight into a message class. When the
#  sender's wire format is wrong the failure is rarely an exception - CDR is self-describing only
#  in the sense that lengths are read from the stream, so a misaligned field turns a vertex count
#  into a nonsense integer and the message either allocates absurdly or arrives full of garbage
#  that looks structurally valid. These helpers make the bytes and the parsed result both visible.
#
#  Off by default. Enable with:  export ROS_TCP_DEBUG_MSGS=1

import os

# ROS 2 wraps every message in a 4-byte CDR encapsulation header. Unity's ROS-TCP-Connector only
# writes it when the ROS2 scripting define is set; a build without it sends ROS 1 bytes that begin
# directly with field data, which is the single most common cause of "the message arrived but the
# contents are nonsense".
_CDR_HEADERS = {
    b"\x00\x01\x00\x00": "CDR_LE - ROS 2 little endian (expected)",
    b"\x00\x00\x00\x00": "CDR_BE - ROS 2 big endian",
    b"\x00\x03\x00\x00": "CDR2_LE",
    b"\x00\x02\x00\x00": "PL_CDR_LE (parameter list)",
}


def enabled():
    return os.environ.get("ROS_TCP_DEBUG_MSGS", "") not in ("", "0", "false", "False")


def cdr_verdict(data):
    """One line on whether these bytes even start like a ROS 2 message."""
    if len(data) < 4:
        return "payload shorter than a CDR header ({} bytes)".format(len(data))
    header = bytes(data[:4])
    known = _CDR_HEADERS.get(header)
    if known:
        return "header {} -> {}".format(header.hex(" "), known)
    return (
        "header {} -> NOT a CDR encapsulation header. The sender is almost certainly in ROS 1 "
        "mode (Unity: Robotics > ROS Settings > Protocol > ROS2, with the target platform active, "
        "then rebuild)".format(header.hex(" "))
    )


def hexdump(data, limit=64):
    head = bytes(data[:limit])
    text = " ".join("{:02x}".format(b) for b in head)
    if len(data) > limit:
        text += " ... (+{} bytes)".format(len(data) - limit)
    return text


def _is_message(value):
    return hasattr(value, "get_fields_and_field_types")


def describe(msg, max_seq=3, _depth=0, _max_depth=6):
    """
    Structural summary of a message: every scalar, and for sequences the length plus the first few
    entries. Deliberately not a full dump - a room-sized shape_msgs/Mesh is tens of thousands of
    vertices, and the length is the number that tells you whether the decode went wrong anyway.
    """
    pad = "  " * _depth
    if _depth > _max_depth:
        return pad + "...\n"

    out = []
    for name, ftype in msg.get_fields_and_field_types().items():
        value = getattr(msg, name)

        if _is_message(value):
            out.append("{}{}: {}\n".format(pad, name, ftype))
            out.append(describe(value, max_seq, _depth + 1, _max_depth))
            continue

        if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
            out.append("{}{}: {} = {!r}\n".format(pad, name, ftype, value))
            continue

        count = len(value)
        out.append("{}{}: {} len={}\n".format(pad, name, ftype, count))
        for index, item in enumerate(list(value)[:max_seq]):
            if _is_message(item):
                out.append("{}  [{}]\n".format(pad, index))
                out.append(describe(item, max_seq, _depth + 2, _max_depth))
            else:
                out.append("{}  [{}] = {!r}\n".format(pad, index, item))
        if count > max_seq:
            out.append("{}  ... {} more\n".format(pad, count - max_seq))

    return "".join(out)


def report(logger, label, data, message):
    """Called after a successful deserialize. Silent unless ROS_TCP_DEBUG_MSGS is set."""
    if not enabled():
        return
    logger.info(
        "\n=== {} ===\n{} bytes, {}\nraw: {}\n{}".format(
            label, len(data), cdr_verdict(data), hexdump(data), describe(message)
        )
    )


def report_failure(logger, label, message_type, data, error):
    """
    Called when deserialize raised. Always logs, regardless of the env var - bytes that could not
    be decoded at all are exactly the ones worth keeping, and they are otherwise lost.
    """
    logger.error(
        "\n=== {} FAILED TO DESERIALIZE as {} ===\n{}: {}\n{} bytes, {}\nraw: {}".format(
            label,
            getattr(message_type, "__name__", message_type),
            type(error).__name__,
            error,
            len(data),
            cdr_verdict(data),
            hexdump(data, limit=128),
        )
    )
