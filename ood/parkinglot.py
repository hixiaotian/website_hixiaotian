from enum import Enum
from abc import ABC
import datetime


class ParkingStatus(Enum):
    empty = "EMPTY"
    occupied = "OCCUPIED"


class SlotSize(Enum):
    small = "SMALL"
    medium = "MEDIUM"
    large = "LARGE"


class VehicleType(Enum):
    motorcycle = "MOTORCYCLE"
    compact = "COMPACT"
    truck = "TRUCK"


class Vehicle(ABC):
    def __init__(self, id: str, vehicle_type: VehicleType):
        self.id = id
        self.vehicle_type = vehicle_type


class Motorcycle(Vehicle):
    def __init__(self, id: str):
        super().__init__(id, VehicleType.motorcycle)


class Compact(Vehicle):
    def __init__(self, id: str):
        super().__init__(id, VehicleType.compact)


class Truck(Vehicle):
    def __init__(self, id: str):
        super().__init__(id, VehicleType.truck)


class ParkingSpot:
    def __init__(self, id: str, status: ParkingStatus, size: SlotSize):
        self.id = id
        self.status = status
        self.size = size
        self.vehicle = None
        self.occupied_time = None

    def fit_size(self, vehicle: Vehicle):
        if self.size == SlotSize.small:
            return vehicle.vehicle_type == VehicleType.motorcycle
        elif self.size == SlotSize.medium:
            return vehicle in [VehicleType.motorcycle, VehicleType.compact]
        elif self.size == SlotSize.large:
            return True
        else:
            return False

    def park_vehicle(self, vehicle: Vehicle):
        if self.status == ParkingStatus.occupied:
            raise Exception("Parking spot is occupied")

        if not self.fit_size(vehicle):
            raise Exception("Vehicle does not fit in this spot")

        self.vehicle = vehicle
        self.status = ParkingStatus.occupied
        self.occupied_time = datetime.now()

    def remove_vehicle(self):
        if self.status == ParkingStatus.empty:
            raise Exception("Parking spot is already empty")

        self.vehicle = None
        self.status = ParkingStatus.emptyx


class Floor:
    def __init__(self, name: str, parking_slots: list[ParkingSpot]):
        self.name = name
        self.parking_spots = parking_slots

    def empty_floor(self):
        for parking_spot in self.parking_spots:
            parking_spot.remove_vehicle()


class ParkingLotController:
    def __init__(self, name: str, floors: list[Floor]):
        self.name = name
        self.floors = floors

    def park_vehicle(self, vehicle: Vehicle):
        for floor in self.floors:
            for parking_spot in floor.parking_spots:
                if parking_spot.fit_size(vehicle):
                    parking_spot.park_vehicle(vehicle)
                    return parking_spot
        raise Exception("No parking spot available")

    def exit_vehicle(self, parking_spot: ParkingSpot):
        self.calculate_fee(parking_spot)
        parking_spot.remove_vehicle()

    def calculate_fee(self, parking_spot: ParkingSpot):
        current_time = datetime.now()
        occupied_time = parking_spot.occupied_time
        duration = (current_time - occupied_time) // 3600
        if duration < 1:
            return 0
        elif duration < 2:
            return 1
        else:
            return 2
