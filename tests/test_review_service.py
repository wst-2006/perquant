import pytest

from src.quant_trading.services.review import ReviewService


def test_review_requires_title_and_content():
    service = ReviewService()
    with pytest.raises(ValueError):
        service.add_entry(title="", content="content")
    with pytest.raises(ValueError):
        service.add_entry(title="title", content="")
