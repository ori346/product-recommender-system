"""
Helper functions for testing. If complex logic is needed, add the function here in this file.
"""

def validate_product_list(response, expected_length=None, required_fields=None):
    """
    Validate that response is a list of product dictionaries with required fields

    Args:
        response: The HTTP response object
        expected_length: Expected number of items
        required_fields: List of required fields in each item (optional)

    Raises:
        AssertionError: If validation fails
    """
    data = response.json()

    # Check if response is a list
    assert isinstance(data, list), f"Response is not a list, got: {type(data)}"
    if expected_length:
        assert len(data) == expected_length, f"Expected {expected_length} items, got {len(data)}"
    assert all(isinstance(item, dict) for item in data), "All items must be dictionaries"

    # Fix the syntax error - check required fields if provided
    if required_fields:
        for item in data:
            assert all(field in item for field in required_fields), f"Item missing required fields: {item}"

    print(f"✅ Validated list of {len(data)} product items with fields: {required_fields}")

def validate_reviews_list(response):
    """
    Validate that the response is a list of review dictionaries

    Args:
        response: The HTTP response object

    Raises:
        AssertionError: If validation fails
    """
    data = response.json()

    # Check if response is a list
    assert isinstance(data, list), f"Response is not a list, got: {type(data)}"

    # Check that all items are dictionaries
    assert all(isinstance(item, dict) for item in data), "All items must be dictionaries"

    # Check that each review has the required fields
    required_fields = ["id", "productId", "userId", "rating", "title", "comment", "created_at"]
    for i, item in enumerate(data):
        for field in required_fields:
            assert field in item, f"Review {i} missing required field: {field}"

    print(f"✅ Validated list of {len(data)} review items")
    return data

def validate_review_exists_by_fields(response, productId, userId, rating, title, comment):
    """
    Validate that a review with the specified fields exists in the response list

    Args:
        response: The HTTP response object
        productId: Expected product ID
        userId: Expected user ID
        rating: Expected rating
        title: Expected title
        comment: Expected comment
    """
    data = response.json()

    # Check if response is a list
    assert isinstance(data, list), f"Response is not a list, got: {type(data)}"

    # Check that all items are dictionaries
    assert all(isinstance(item, dict) for item in data), "All items must be dictionaries"

    # Look for the expected review in the list
    found_review = None
    for i, review in enumerate(data):
        # Convert rating to int for comparison (handle string/int mismatch)
        review_rating = review.get('rating')
        expected_rating = int(rating) if isinstance(rating, str) else rating

        if (review.get('productId') == productId and
            review.get('userId') == userId and
            review_rating == expected_rating and
            review.get('title') == title and
            review.get('comment') == comment):
            found_review = review
            break

    assert found_review is not None, f"Expected review not found in response. Looking for productId={productId}, userId={userId}, rating={rating}, title={title}, comment={comment}"
    print(f"✅ Found expected review with ID {found_review['id']}")

def validate_review_exists(response, expected_review):
    """
    Validate that the expected review exists in the response list

    Args:
        response: The HTTP response object
        expected_review: The review that should exist in the list
    """
    data = response.json()

    # Check if response is a list
    assert isinstance(data, list), f"Response is not a list, got: {type(data)}"

    # Check that all items are dictionaries
    assert all(isinstance(item, dict) for item in data), "All items must be dictionaries"

    # Look for the expected review in the list
    found_review = None
    for review in data:
        if (review.get('productId') == expected_review.get('productId') and
            review.get('userId') == expected_review.get('userId') and
            review.get('rating') == expected_review.get('rating') and
            review.get('title') == expected_review.get('title') and
            review.get('comment') == expected_review.get('comment')):
            found_review = review
            break

    assert found_review is not None, f"Expected review not found in response. Looking for: {expected_review}"
    print(f"✅ Found expected review with ID {found_review['id']}")

def validate_review_added(response, reviews, review):
    """
    Validate that the response is add to the end of the list

    Args:
        response: The HTTP response object
        reviews: List of reviews
    """
    data = response.json()
    assert data[1:] == reviews, f"there is a difference between the first {len(reviews)} reviews in the response and the reviews in the list"
    assert data[0] == review, f"the last review in the response is not the same as the response"
