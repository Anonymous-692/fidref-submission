#!/usr/bin/env python3
"""Minimal handcrafted fixtures for the tau-bench retail domain slice."""

from __future__ import annotations

import copy
from typing import Any, Dict


def get_initial_data() -> Dict[str, Any]:
    """Return a fresh copy of the handcrafted retail database slice.

    Schema matches sierra-research/tau-bench retail domain:
      - data['users']: user profiles with payment methods and order history
      - data['orders']: delivered and pending orders with item details and payment history
      - data['products']: catalog with product variants, prices, and availability
    """
    return {
        "users": {
            "user_1": {
                "user_id": "user_1",
                "name": {
                    "first_name": "Alice",
                    "last_name": "Smith",
                },
                "email": "alice.smith@example.com",
                "address": {
                    "address1": "123 Market Street",
                    "address2": "Apt 4B",
                    "city": "Seattle",
                    "state": "WA",
                    "country": "USA",
                    "zip": "98101",
                },
                "payment_methods": {
                    "gift_card_0": {
                        "id": "gift_card_0",
                        "source": "gift_card",
                        "balance": 15.0,
                    },
                    "credit_card_0": {
                        "id": "credit_card_0",
                        "source": "credit_card",
                        "brand": "visa",
                        "last_four": "4242",
                    },
                },
                "orders": ["#W1", "#W2"],
            }
        },
        "orders": {
            "#W1": {
                "order_id": "#W1",
                "user_id": "user_1",
                "status": "delivered",
                "items": [
                    {
                        "name": "Running Shoes",
                        "product_id": "prod_shoes",
                        "item_id": "shoe_black_9",
                        "price": 100.0,
                        "options": {"color": "black", "size": "9"},
                    },
                    {
                        "name": "Cotton T-Shirt",
                        "product_id": "prod_shirt",
                        "item_id": "shirt_blue_m",
                        "price": 40.0,
                        "options": {"color": "blue", "size": "M"},
                    },
                ],
                "fulfillments": [
                    {
                        "tracking_id": ["TRK12345678"],
                        "item_ids": ["shoe_black_9", "shirt_blue_m"],
                    }
                ],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 140.0,
                        "payment_method_id": "credit_card_0",
                    }
                ],
            },
            "#W2": {
                "order_id": "#W2",
                "user_id": "user_1",
                "status": "pending",
                "items": [
                    {
                        "name": "Wool Socks",
                        "product_id": "prod_socks",
                        "item_id": "socks_white",
                        "price": 25.0,
                        "options": {"color": "white"},
                    }
                ],
                "fulfillments": [],
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "amount": 25.0,
                        "payment_method_id": "gift_card_0",
                    }
                ],
            },
        },
        "products": {
            "prod_shoes": {
                "product_id": "prod_shoes",
                "name": "Running Shoes",
                "variants": {
                    "shoe_black_9": {
                        "item_id": "shoe_black_9",
                        "options": {"color": "black", "size": "9"},
                        "available": True,
                        "price": 100.0,
                    },
                    "shoe_black_10": {
                        "item_id": "shoe_black_10",
                        "options": {"color": "black", "size": "10"},
                        "available": True,
                        "price": 100.0,
                    },
                    "shoe_red_9": {
                        "item_id": "shoe_red_9",
                        "options": {"color": "red", "size": "9"},
                        "available": True,
                        "price": 130.0,
                    },
                    "shoe_blue_9": {
                        "item_id": "shoe_blue_9",
                        "options": {"color": "blue", "size": "9"},
                        "available": False,
                        "price": 100.0,
                    },
                },
            },
            "prod_shirt": {
                "product_id": "prod_shirt",
                "name": "Cotton T-Shirt",
                "variants": {
                    "shirt_blue_m": {
                        "item_id": "shirt_blue_m",
                        "options": {"color": "blue", "size": "M"},
                        "available": True,
                        "price": 40.0,
                    },
                    "shirt_blue_l": {
                        "item_id": "shirt_blue_l",
                        "options": {"color": "blue", "size": "L"},
                        "available": True,
                        "price": 60.0,
                    },
                },
            },
            "prod_socks": {
                "product_id": "prod_socks",
                "name": "Wool Socks",
                "variants": {
                    "socks_white": {
                        "item_id": "socks_white",
                        "options": {"color": "white"},
                        "available": True,
                        "price": 25.0,
                    },
                },
            },
        },
    }


def make_data_copy() -> Dict[str, Any]:
    """Create a deep copy of initial data for safe mutation during simulation."""
    return copy.deepcopy(get_initial_data())
