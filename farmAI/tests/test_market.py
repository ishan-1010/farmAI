import json
import pytest
from unittest.mock import patch, MagicMock
from agents.market import get_market_advice, _lookup_price

MOCK_MARKET = {
    "crop": "Tomato",
    "modal_price": 1000,
    "price_unit": "per Quintal",
    "market": "Pune, Maharashtra",
    "last_updated": "2025-05-01",
    "recommendation": "SELL NOW",
    "reasoning": "Price is at seasonal high and disease severity is moderate — sell before quality degrades.",
    "price_trend": "stable",
}


def _make_mock_client(response_dict):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(response_dict)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


class TestLookupPrice:
    def test_tomato_found(self):
        result = _lookup_price("Tomato")
        assert result is not None
        assert result["crop"] == "Tomato"
        assert result["modal_price"] == 1000

    def test_case_insensitive_match(self):
        result_lower = _lookup_price("tomato")
        result_upper = _lookup_price("TOMATO")
        assert result_lower is not None
        assert result_upper is not None
        assert result_lower["modal_price"] == result_upper["modal_price"]

    def test_unknown_crop_returns_none(self):
        result = _lookup_price("Dragonfruit")
        assert result is None

    def test_rice_found(self):
        result = _lookup_price("Rice")
        assert result is not None
        assert result["modal_price"] == 2200

    def test_partial_match_works(self):
        result = _lookup_price("Mang")  # partial match for Mango
        assert result is not None


class TestGetMarketAdvice:
    @patch("agents.market.Groq")
    def test_schema_all_keys_present(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_MARKET)
        result = get_market_advice("Tomato", 3, 21)
        for key in ["crop", "modal_price", "price_unit", "market", "last_updated", "recommendation", "reasoning", "price_trend"]:
            assert key in result, f"Missing key: {key}"

    @patch("agents.market.Groq")
    def test_price_from_csv_not_llm(self, mock_groq):
        # LLM might hallucinate price — CSV values must override
        mock_groq.return_value = _make_mock_client({**MOCK_MARKET, "modal_price": 9999})
        result = get_market_advice("Tomato", 3, 21)
        assert result["modal_price"] == 1000  # from CSV, not LLM

    @patch("agents.market.Groq")
    def test_unknown_crop_no_crash(self, mock_groq):
        result = get_market_advice("Dragonfruit", 2, 14)
        assert result["recommendation"] == "CONSULT LOCAL MANDI"
        assert result["modal_price"] == 0
        mock_groq.assert_not_called()  # no API call for unknown crops

    @patch("agents.market.Groq")
    def test_recommendation_valid_value(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_MARKET)
        result = get_market_advice("Tomato", 3, 21)
        assert result["recommendation"] in ["SELL NOW", "WAIT", "PROCESS LOCALLY", "CONSULT LOCAL MANDI"]
