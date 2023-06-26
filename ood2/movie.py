#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 19:09:13 2023

@author: jiayuanhan
"""
# Designing a movie ticket booking system can be broken down into several smaller components that are easier to handle. Here is a high-level design for such a system.

# ## Basic Classes

# 1. **Movie**: This class represents a movie in the theater. It can contain properties such as title, duration, rating, and genre.

# 2. **Show**: This class represents a particular show of a movie at a given time. It can contain properties such as movie, start time, end time, and a reference to the `Screen` object (the hall where the show will be presented).

# 3. **Screen**: This class represents a cinema hall with a particular seating arrangement. It can contain properties like screen number, total seats, a list of `Seat` objects, and maybe a reference to a `Show` object that will be presented in the hall.

# 4. **Seat**: This class represents a particular seat in a screen. It could have properties like row number, column number, and status (booked or available).

# 5. **Booking**: This class represents a booking made by a user. It can contain properties like booking ID, show details, seats booked, total cost, and the status of the booking (confirmed, pending, cancelled).

# 6. **User**: This class represents a user of the booking system. It can contain properties like user ID, name, contact details, and a list of `Booking` objects made by the user.

# ## Example

# Here is a simple Python code demonstrating these classes:

# ```python
class Movie:
    def __init__(self, title, duration, rating, genre):
        self.title = title
        self.duration = duration
        self.rating = rating
        self.genre = genre

class Show:
    def __init__(self, movie, start_time, end_time, screen):
        self.movie = movie
        self.start_time = start_time
        self.end_time = end_time
        self.screen = screen

class Screen:
    def __init__(self, screen_number, total_seats):
        self.screen_number = screen_number
        self.total_seats = total_seats
        self.seats = self.initialize_seats()

    def initialize_seats(self):
        pass

class Seat:
    def __init__(self, row, column):
        self.row = row
        self.column = column
        self.status = "Available"

class Booking:
    def __init__(self, booking_id, show, seats, total_cost):
        self.booking_id = booking_id
        self.show = show
        self.seats = seats
        self.total_cost = total_cost
        self.status = "Pending"

class User:
    def __init__(self, user_id, name, contact):
        self.user_id = user_id
        self.name = name
        self.contact = contact
        self.bookings = []
# ```

# Please note that the above design is a simple one, and doesn't handle many practical issues like handling concurrent bookings, allowing for different pricing for different shows or seats, providing a user interface, etc. In a real-world scenario, these factors will add considerable complexity to the design.

# Moreover, in a production system, you would want to ensure that you handle transactions correctly to avoid problems like double booking of the same seat. This would likely involve some sort of locking or concurrency control mechanism, which is not shown in this simple example. 

# Similarly, for tracking availability, the system would likely need to query the database to find shows with available seats. This could be done with a simple SQL query in a relational database, or with appropriate data retrieval operations in a non-relational database.