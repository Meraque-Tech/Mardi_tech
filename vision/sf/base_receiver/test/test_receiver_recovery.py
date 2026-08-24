import json

import pytest
from serial import SerialException

from base_receiver.receiver_node import RoverGnssReader


def pvt_line(fix=3):
    return json.dumps(
        {
            "type": "pvt",
            "fix": fix,
            "carr": 0,
            "diff": False,
            "corrAge": 0,
            "lat": 1.958,
            "lon": 103.229,
            "alt": 46.7,
            "hacc": 1.0,
            "vacc": 1.5,
        }
    ).encode("ascii")


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeStopEvent:
    def __init__(self, clock):
        self.clock = clock
        self.stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, timeout):
        self.clock.advance(timeout)
        return self.stopped


class FakeSerial:
    def __init__(self, responses, clock):
        self.responses = list(responses)
        self.clock = clock
        self.closed = False
        self.input_buffer_reset = False

    def reset_input_buffer(self):
        self.input_buffer_reset = True

    def readline(self):
        self.clock.advance(1.0)
        response = self.responses.pop(0) if self.responses else b""
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


class SerialFactory:
    def __init__(self, sessions, clock):
        self.sessions = list(sessions)
        self.clock = clock
        self.connections = []
        self.opened_ports = []

    def __call__(self, port, baud, timeout):
        del baud, timeout
        self.opened_ports.append(port)
        connection = FakeSerial(self.sessions.pop(0), self.clock)
        self.connections.append(connection)
        return connection


def make_reader(sessions, timeout=2.5):
    clock = FakeClock()
    factory = SerialFactory(sessions, clock)
    events = []
    reader = RoverGnssReader(
        "/dev/gnss_rtk",
        115200,
        reconnect_interval=0.5,
        stale_reconnect_timeout=timeout,
        event_callback=lambda level, message: events.append((level, message)),
        serial_factory=factory,
        monotonic=clock.monotonic,
    )
    return reader, FakeStopEvent(clock), factory, events


@pytest.mark.parametrize(
    "stalled_responses",
    [
        [b"", b"", b""],
        [b"not-json", b"not-json", b"not-json"],
        [b'{"type":"pvt"}', b'{"type":"pvt"}', b'{"type":"pvt"}'],
        [b'{"type":"status"}', b'{"type":"status"}', b'{"type":"status"}'],
    ],
)
def test_silent_or_invalid_stream_reopens_and_recovers(stalled_responses):
    reader, stop_event, factory, events = make_reader(
        [stalled_responses, [pvt_line()]]
    )

    recovered = next(
        message
        for message in reader.stream(stop_event)
        if message.get("type") == "pvt"
    )

    assert recovered["type"] == "pvt"
    assert len(factory.connections) == 2
    assert factory.connections[0].closed
    assert factory.connections[1].input_buffer_reset
    assert factory.opened_ports == ["/dev/gnss_rtk", "/dev/gnss_rtk"]
    assert any("stream stale" in message for _level, message in events)
    assert any("stream recovered" in message for _level, message in events)


def test_valid_pvt_then_silence_reopens_the_serial_device():
    reader, stop_event, factory, events = make_reader(
        [[pvt_line(), b"", b"", b""], [pvt_line()]]
    )
    stream = reader.stream(stop_event)

    first = next(stream)
    recovered = next(stream)

    assert first["type"] == "pvt"
    assert recovered["type"] == "pvt"
    assert len(factory.connections) == 2
    assert factory.connections[0].closed
    assert any("stream stale" in message for _level, message in events)


def test_serial_exception_reopens_and_recovers():
    reader, stop_event, factory, events = make_reader(
        [[SerialException("USB endpoint stalled")], [pvt_line()]]
    )

    recovered = next(reader.stream(stop_event))

    assert recovered["type"] == "pvt"
    assert len(factory.connections) == 2
    assert factory.connections[0].closed
    assert any("serial read failed" in message for _level, message in events)
    assert any("stream recovered" in message for _level, message in events)


def test_valid_no_fix_pvt_keeps_connection_alive():
    reader, stop_event, factory, events = make_reader(
        [[pvt_line(fix=0), pvt_line(fix=0)]],
        timeout=1.5,
    )
    stream = reader.stream(stop_event)

    first = next(stream)
    second = next(stream)

    assert first["fixLabel"] == "NO_FIX"
    assert second["fixLabel"] == "NO_FIX"
    assert len(factory.connections) == 1
    assert not factory.connections[0].closed
    assert not any("stream stale" in message for _level, message in events)
