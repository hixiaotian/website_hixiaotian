#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 20:18:29 2023

@author: jiayuanhan
"""
# Designing an Airline Reservation System involves creating different classes to represent various entities in the system, such as Flights, Passengers, Tickets, and Reservations. Let's see a high-level design of such a system:

# ## Basic Classes

# 1. **Flight**: This class represents a flight. It can have properties such as flight number, departure airport, arrival airport, departure time, arrival time, total seat count, and a seat map which can hold the status (booked or available) of each seat.

# 2. **Passenger**: This class represents a passenger. It can have properties like passenger ID, name, contact information, and a list of tickets booked.

# 3. **Ticket**: This class represents a ticket. It can have properties like ticket ID, the passenger who holds the ticket, the flight it corresponds to, the seat number, and the status (confirmed or cancelled).

# 4. **Reservation**: This class represents a reservation. It can have properties like reservation ID, the passenger who made the reservation, the list of flights, and the status (paid or unpaid).

# Here is a Python example that demonstrates these classes:

# ```python
class Flight:
    def __init__(self, flight_number, departure_airport, arrival_airport, departure_time, arrival_time, total_seats):
        self.flight_number = flight_number
        self.departure_airport = departure_airport
        self.arrival_airport = arrival_airport
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.total_seats = total_seats
        self.seat_map = ['Available'] * total_seats

class Passenger:
    def __init__(self, passenger_id, name, contact):
        self.passenger_id = passenger_id
        self.name = name
        self.contact = contact
        self.tickets = []

class Ticket:
    def __init__(self, ticket_id, passenger, flight, seat_number):
        self.ticket_id = ticket_id
        self.passenger = passenger
        self.flight = flight
        self.seat_number = seat_number
        self.status = "Confirmed"

class Reservation:
    def __init__(self, reservation_id, passenger, flights):
        self.reservation_id = reservation_id
        self.passenger = passenger
        self.flights = flights
        self.status = "Unpaid"

    def make_payment(self):
        pass
# ```

# This is a high-level and simplified design that doesn't include many practical considerations like error handling (what if a flight is full, or a passenger tries to book a flight that has already departed?), or database interactions. Also, for checking seat availability and flight schedules, you might want to add methods to the `Flight` class. However, this basic design provides a good starting point to build upon.