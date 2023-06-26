#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 18:46:48 2023

@author: jiayuanhan
"""
# Designing a parking lot system is a common object-oriented design question. This problem can be broken down into several smaller problems that need to be addressed:

# 1. **Handling different types of vehicles**: This could be solved using a simple class hierarchy. You could have a base `Vehicle` class and subclasses for different types of vehicles like `Car`, `Truck`, `Motorcycle`, etc. These subclasses may have different properties or methods, depending on the specific rules of the parking lot.

# 2. **Multiple levels**: To handle this, you could have a `Level` class. Each `Level` would have a list of `ParkingSpot` objects. A `ParkingLot` would then consist of multiple `Level` objects.

# 3. **Tracking available spots**: Each `ParkingSpot` could have a `isAvailable` method that tells whether the spot is currently occupied or not. The `Level` could have a method `findAvailableSpot` that iterates over its `ParkingSpot` objects and returns an available one. Similarly, `ParkingLot` could have a method `findAvailableSpot` that iterates over its `Level` objects and uses their `findAvailableSpot` method to find an available spot.

# Here is a simple implementation in Python:

# ```python
class Vehicle:
    def __init__(self, license_plate):
        self.license_plate = license_plate

class Car(Vehicle):
    pass

class Truck(Vehicle):
    pass

class Motorcycle(Vehicle):
    pass

class ParkingSpot:
    def __init__(self):
        self.vehicle = None

    def park(self, vehicle):
        self.vehicle = vehicle

    def leave(self):
        self.vehicle = None

    def is_available(self):
        return self.vehicle is None

class Level:
    def __init__(self, num_spots):
        self.spots = [ParkingSpot() for _ in range(num_spots)]

    def find_available_spot(self):
        for spot in self.spots:
            if spot.is_available():
                return spot
        return None

class ParkingLot:
    def __init__(self, num_levels, num_spots_per_level):
        self.levels = [Level(num_spots_per_level) for _ in range(num_levels)]

    def find_available_spot(self):
        for level in self.levels:
            spot = level.find_available_spot()
            if not spot.available():
                return spot
        return None
# ```

# This design is quite simple and doesn't take into account many complexities of a real-world parking lot system. For example, it doesn't handle the case where a truck needs a larger spot, or where a motorcycle can fit in a smaller spot. It also doesn't handle the case where the parking lot charges different rates for different vehicle types or different times of day. But it's a starting point, and it illustrates some key object-oriented design principles.

# For a more comprehensive solution, you'd likely need to take into account more specific requirements and rules of the parking lot, and design your classes and their interactions accordingly.