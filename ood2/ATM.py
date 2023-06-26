#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 19:10:53 2023

@author: jiayuanhan
"""
# Designing an ATM system involves thinking about the different components that make up the system. This includes the user interface, the network connections to bank systems, and the hardware that handles cash and cards. Here's a simplified model of what such a system might look like, focusing primarily on the software aspects:

# ## Basic Classes

# 1. **Account**: This class would represent a bank account. It can contain properties such as account number, account holder's name, account type (savings, current, etc.), and balance.

# 2. **Card**: This class would represent an ATM card. It can contain properties like card number, account number (linked to the `Account` object), cardholder's name, card expiry date, and PIN.

# 3. **Transaction**: This class would represent a bank transaction. It can contain properties like transaction ID, account number, transaction type (withdrawal, deposit, transfer, balance inquiry), transaction amount, and transaction date/time.

# 4. **ATM**: This class would represent the ATM machine itself. It can contain properties like ATM ID, location, cash available, and a list of `Transaction` objects. It might also contain methods for handling transactions, like `handleWithdrawal()`, `handleDeposit()`, `handleTransfer()`, and `handleBalanceInquiry()`.

# 5. **Bank**: This class would represent the bank itself. It might contain a list of `Account` objects, a list of `Card` objects, and methods for validating cards and transactions.

# ## Example

# Here's a high-level example in Python, without method implementation:

# ```python
class Account:
    def __init__(self, account_number, account_holder, account_type, balance):
        self.account_number = account_number
        self.account_holder = account_holder
        self.account_type = account_type
        self.balance = balance

class Card:
    def __init__(self, card_number, account_number, card_holder, expiry_date, pin):
        self.card_number = card_number
        self.account_number = account_number
        self.card_holder = card_holder
        self.expiry_date = expiry_date
        self.pin = pin

class Transaction:
    def __init__(self, transaction_id, account_number, transaction_type, amount, date_time):
        self.transaction_id = transaction_id
        self.account_number = account_number
        self.transaction_type = transaction_type
        self.amount = amount
        self.date_time = date_time

class ATM:
    def __init__(self, atm_id, location, cash_available):
        self.atm_id = atm_id
        self.location = location
        self.cash_available = cash_available
        self.transactions = []

    def handle_withdrawal(self, card, pin, amount):
        pass

    def handle_deposit(self, card, pin, amount):
        pass

    def handle_transfer(self, card, pin, amount, target_account_number):
        pass

    def handle_balance_inquiry(self, card, pin):
        pass

class Bank:
    def __init__(self):
        self.accounts = []
        self.cards = []

    def validate_card(self, card, pin):
        pass

    def validate_transaction(self, transaction):
        pass
# ```

# Note that this is a simplified design, and doesn't handle many real-world factors like encryption for card PINs, connection to a central bank server, handling multiple concurrent transactions, dispensing different denominations of bills, etc. Also, error checking, such as insufficient balance, incorrect pin, card expiry, etc., have not been included in this design. A real-world ATM system would be significantly more complex.