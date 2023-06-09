from enum import Enum
from heapq import heappush, heappop


class Direction(Enum):
    up = "UP"
    down = "DOWN"
    idle = "IDLE"


class RequestType(Enum):
    external = "EXTERNAL"
    internal = "INTERNAL"


class Request:
    def __init__(
            self, origin: int, target: int, request_type: RequestType,
            direction: Direction):
        self.origin = origin
        self.target = target
        self.request_type = request_type
        self.direction = direction


class Button:
    def __init__(self, floor: int):
        self.floor = floor


class Elevator:
    def __init__(self, current_floor: int):
        self.direction = Direction.idle
        self.current_floor = current_floor
        self.up_stops = []
        self.down_stops = []
        print(f"Elevator starts at floor {current_floor}")

    def send_up_request(self, up_request: Request):
        if up_request.request_type == RequestType.external:
            heappush(self.up_stops, (up_request.origin, up_request.origin))
        heappush(self.up_stops, (up_request.target, up_request.origin))

    def send_down_request(self, down_request: Request):
        if down_request.request_type == RequestType.external:
            heappush(self.down_stops, (-down_request.origin, down_request.origin))
        heappush(self.down_stops, (-down_request.target, down_request.origin))

    def run(self):
        while self.up_stops or self.down_stops:
            self.process_requests()

    def process_requests(self):
        if self.direction in [Direction.up, Direction.idle]:
            self.process_up_requests()
            self.process_down_requests()

        else:
            self.process_down_requests()
            self.process_up_requests()

    def process_up_requests(self):
        while self.up_stops:
            target, origin = heappop(self.up_stops)

            self.current_floor = target

            if target == origin:
                print(f"Stopping at floor {target} to pick up passenger")
            else:
                print(f"Stopping at floor {target} to drop off passenger")

            if self.down_stops:
                self.direction = Direction.down
            else:
                self.direction = Direction.idle

    def process_down_requests(self):
        while self.down_stops:
            target, origin = heappop(self.down_stops)

            self.current_floor = -target

            if target == -origin:
                print(f"Stopping at floor {-target} to pick up passenger")
            else:
                print(f"Stopping at floor {-target} to drop off passenger")

            if self.up_stops:
                self.direction = Direction.up
            else:
                self.direction = Direction.idle


elevator = Elevator(0)
up_request = Request(elevator.current_floor, 5,
                     RequestType.internal, Direction.up)
up_request2 = Request(elevator.current_floor, 3,
                      RequestType.internal, Direction.up)

down_request = Request(elevator.current_floor, 1,
                       RequestType.internal, Direction.down)
down_request2 = Request(elevator.current_floor, 4,
                        RequestType.internal, Direction.down)

up_request3 = Request(2, 6, RequestType.external, Direction.up)
down_request3 = Request(15, 3, RequestType.external, Direction.down)

elevator.send_up_request(up_request)
elevator.send_up_request(up_request2)
elevator.send_down_request(down_request)
elevator.send_down_request(down_request2)
elevator.send_up_request(up_request3)
elevator.send_down_request(down_request3)

elevator.run()
