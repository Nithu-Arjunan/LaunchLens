import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests


BACKEND_DIR = Path(__file__).resolve().parents[1] / "src" / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class AmazonApiModelTests(unittest.TestCase):
    def test_fetch_amazon_search_returns_pydantic_shaped_dict(self):
        api = importlib.import_module("api")
        payload = {
            "results": [
                {
                    "content": {
                        "results": {
                            "organic": [
                                {
                                    "title": "Eco Lunch Box",
                                    "price": "$19.99",
                                    "currency": "USD",
                                    "rating": 4.6,
                                    "reviews_count": 128,
                                    "url": "https://example.com/item",
                                    "asin": "B001",
                                }
                            ]
                        }
                    }
                }
            ]
        }

        with patch.dict(
            os.environ,
            {"OXYLABS_USERNAME": "user", "OXYLABS_PASSWORD": "pass"},
            clear=True,
        ):
            with patch("requests.post", return_value=_Response(payload)):
                result = api.fetch_amazon_search(
                    "eco-friendly lunch boxes",
                    max_results=3,
                    region="United Arab Emirates",
                )

        self.assertEqual(result["source"], "amazon_search")
        self.assertEqual(result["query"], "eco-friendly lunch boxes")
        self.assertEqual(result["region"], "United Arab Emirates")
        self.assertEqual(result["domain"], "ae")
        self.assertEqual(result["max_results"], 3)
        self.assertEqual(result["listings"][0]["title"], "Eco Lunch Box")
        self.assertEqual(result["listings"][0]["rating_count"], 128)
        self.assertNotIn("error", result)

    def test_fetch_amazon_search_connection_error_returns_model_shaped_degraded_result(self):
        api = importlib.import_module("api")

        with patch.dict(
            os.environ,
            {"OXYLABS_USERNAME": "user", "OXYLABS_PASSWORD": "pass"},
            clear=True,
        ):
            with patch("requests.post", side_effect=requests.ConnectionError("reset")):
                result = api.fetch_amazon_search("smart mugs", region="India")

        self.assertEqual(result["source"], "amazon_search")
        self.assertEqual(result["query"], "smart mugs")
        self.assertEqual(result["region"], "India")
        self.assertEqual(result["domain"], "in")
        self.assertEqual(result["listings"], [])
        self.assertIn("ConnectionError: reset", result["error"])

    def test_fetch_amazon_product_returns_pydantic_shaped_dict(self):
        api = importlib.import_module("api")
        payload = {
            "results": [
                {
                    "content": {
                        "title": "Eco Lunch Box",
                        "asin": "B123",
                        "brand": "EcoBrand",
                        "price": "AED 49.99",
                        "currency": "AED",
                        "rating": 4.7,
                        "reviews_count": 321,
                        "availability": "In Stock",
                        "description": "Reusable lunch box.",
                        "features": ["Reusable", "BPA free"],
                        "images": ["https://example.com/image.jpg"],
                        "categories": ["Kitchen", "Lunch Boxes"],
                        "url": "https://example.com/product",
                    }
                }
            ]
        }

        with patch.dict(
            os.environ,
            {"OXYLABS_USERNAME": "user", "OXYLABS_PASSWORD": "pass"},
            clear=True,
        ):
            with patch("requests.post", return_value=_Response(payload)) as post:
                result = api.fetch_amazon_product(
                    "B123",
                    region="United Arab Emirates",
                    domain="ae",
                )

        body = post.call_args.kwargs["json"]
        self.assertEqual(body["source"], "amazon_product")
        self.assertEqual(body["query"], "B123")
        self.assertEqual(body["domain"], "ae")
        self.assertEqual(result["source"], "amazon_product")
        self.assertEqual(result["query"], "B123")
        self.assertEqual(result["region"], "United Arab Emirates")
        self.assertEqual(result["product"]["title"], "Eco Lunch Box")
        self.assertEqual(result["product"]["asin"], "B123")
        self.assertEqual(result["product"]["bullet_points"], ["Reusable", "BPA free"])
        self.assertNotIn("error", result)

    def test_fetch_amazon_product_normalizes_list_description(self):
        api = importlib.import_module("api")
        payload = {
            "results": [
                {
                    "content": {
                        "title": "Eco Lunch Box",
                        "asin": "B123",
                        "description": [
                            "https://m.media-amazon.com/images/I/image1.jpg",
                            "https://m.media-amazon.com/images/I/image2.jpg",
                        ],
                    }
                }
            ]
        }

        with patch.dict(
            os.environ,
            {"OXYLABS_USERNAME": "user", "OXYLABS_PASSWORD": "pass"},
            clear=True,
        ):
            with patch("requests.post", return_value=_Response(payload)):
                result = api.fetch_amazon_product("B123", region="United Arab Emirates")

        self.assertEqual(
            result["product"]["description"],
            "https://m.media-amazon.com/images/I/image1.jpg\n"
            "https://m.media-amazon.com/images/I/image2.jpg",
        )

    def test_fetch_amazon_product_connection_error_returns_model_shaped_degraded_result(self):
        api = importlib.import_module("api")

        with patch.dict(
            os.environ,
            {"OXYLABS_USERNAME": "user", "OXYLABS_PASSWORD": "pass"},
            clear=True,
        ):
            with patch("requests.post", side_effect=requests.ConnectionError("reset")):
                result = api.fetch_amazon_product("B123", region="India")

        self.assertEqual(result["source"], "amazon_product")
        self.assertEqual(result["query"], "B123")
        self.assertEqual(result["region"], "India")
        self.assertEqual(result["domain"], "in")
        self.assertIn("ConnectionError: reset", result["error"])


if __name__ == "__main__":
    unittest.main()
