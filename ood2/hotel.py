#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 19:10:52 2023

@author: jiayuanhan
"""
# Designing a Hotel Management System involves creating different classes for handling various aspects of the hotel operations. Here is a high-level design for such a system:

# ## Basic Classes

# 1. **Room**: This class represents a room in the hotel. It can contain properties like room number, room type (single, double, suite), status (available, occupied, not available), and rate.

# 2. **Reservation**: This class represents a booking made by a customer. It can contain properties like reservation ID, customer details, check-in date, check-out date, room assigned, and payment status.

# 3. **Customer**: This class represents a customer. It can contain properties like customer ID, name, contact details, and a list of `Reservation` objects related to the customer.

# 4. **Billing**: This class represents a bill. It can contain properties like bill ID, reservation ID, room charges, other charges (like food, laundry, etc.), total amount, and payment status.

# 5. **Hotel**: This class represents the hotel itself. It can contain a list of `Room` objects, a list of `Reservation` objects, a list of `Customer` objects, a list of `Billing` objects, and methods to handle operations like making a reservation, checking in a customer, checking out a customer, and generating a bill.

# ## Example

# Here is a basic Python code demonstrating these classes:

# ```python
class Room:
    def __init__(self, room_number, room_type, rate):
        self.room_number = room_number
        self.room_type = room_type
        self.status = "Available"
        self.rate = rate

class Reservation:
    def __init__(self, reservation_id, customer, check_in_date, check_out_date, room):
        self.reservation_id = reservation_id
        self.customer = customer
        self.check_in_date = check_in_date
        self.check_out_date = check_out_date
        self.room = room
        self.payment_status = "Pending"

class Customer:
    def __init__(self, customer_id, name, contact):
        self.customer_id = customer_id
        self.name = name
        self.contact = contact
        self.reservations = []

class Billing:
    def __init__(self, bill_id, reservation, room_charges, other_charges):
        self.bill_id = bill_id
        self.reservation = reservation
        self.room_charges = room_charges
        self.other_charges = other_charges
        self.total_amount = self.room_charges + self.other_charges
        self.payment_status = "Unpaid"

class Hotel:
    def __init__(self):
        self.rooms = []
        self.reservations = []
        self.customers = []
        self.bills = []

    def make_reservation(self, customer, check_in_date, check_out_date, room_type):
        pass

    def check_in_customer(self, reservation_id):
        pass

    def check_out_customer(self, reservation_id):
        pass

    def generate_bill(self, reservation_id):
        pass
# ```

# Please note that the above design is a simplified one, and doesn't handle many practical issues like handling concurrent reservations, allowing for different pricing for different rooms or seasons, providing a user interface, etc. In a real-world scenario, these factors will add considerable complexity to the design.

# Moreover, in a production system, you would want to ensure that you handle transactions correctly to avoid problems like double booking of the same room. This would likely involve some sort of locking or concurrency control mechanism, which is not shown in this simple example. 

# Similarly, for tracking room availability, the system would likely need to query the database to find rooms with "Available" status. This could be done with a simple SQL query