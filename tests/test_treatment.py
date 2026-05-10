import json
import pytest
from unittest.mock import patch, MagicMock
from agents.treatment import get_treatment

MOCK_TREATMENT = {
    "immediate_action": "Remove and destroy all infected leaves immediately.",
    "organic_treatment": {
        "method": "Neem oil spray",
        "frequency": "Every 7 days",
        "preparation": "Mix 5ml neem oil in 1L water with a drop of dish soap",
    },
    "chemical_treatment": {
        "pesticide": "Mancozeb 75% WP",
        "dosage": "2g per litre of water",
        "frequency": "Every 10-14 days",
    },
    "prevention": [
        "Avoid overhead watering",
        "Ensure good air circulation between plants",
        "Rotate crops each season",
        "Remove plant debris after harvest",
    ],
    "estimated_recovery_days": 21,
}


def _make_mock_client(response_dict):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(response_dict)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


class TestGetTreatment:
    @patch("agents.treatment.Groq")
    def test_schema_all_keys_present(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_TREATMENT)
        result = get_treatment("Early Blight", 3, "Tomato")
        for key in ["immediate_action", "organic_treatment", "chemical_treatment", "prevention", "estimated_recovery_days"]:
            assert key in result, f"Missing key: {key}"

    @patch("agents.treatment.Groq")
    def test_organic_treatment_has_required_subkeys(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_TREATMENT)
        result = get_treatment("Early Blight", 3, "Tomato")
        organic = result["organic_treatment"]
        assert "method" in organic
        assert "frequency" in organic
        assert "preparation" in organic

    @patch("agents.treatment.Groq")
    def test_prevention_is_list(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_TREATMENT)
        result = get_treatment("Early Blight", 3, "Tomato")
        assert isinstance(result["prevention"], list)
        assert len(result["prevention"]) > 0

    @patch("agents.treatment.Groq")
    def test_recovery_days_is_integer(self, mock_groq):
        mock_groq.return_value = _make_mock_client(MOCK_TREATMENT)
        result = get_treatment("Early Blight", 3, "Tomato")
        assert isinstance(result["estimated_recovery_days"], int)

    @patch("agents.treatment.Groq")
    def test_missing_keys_get_defaults(self, mock_groq):
        mock_groq.return_value = _make_mock_client({"immediate_action": "Act fast"})
        result = get_treatment("Unknown Disease", 2, "Wheat")
        assert result["immediate_action"] == "Act fast"
        assert "organic_treatment" in result
        assert "prevention" in result
        assert result["estimated_recovery_days"] == 14

    @patch("agents.treatment.Groq")
    def test_pest_defaults_use_insecticide(self, mock_groq):
        mock_groq.return_value = _make_mock_client({"immediate_action": "Remove egg masses"})
        result = get_treatment("Fall Armyworm", 4, "Maize", problem_type="pest")
        chem = result["chemical_treatment"]
        # pest defaults must reference insecticide, not fungicide
        assert "pesticide" in chem
        assert "Emamectin" in chem["pesticide"] or "pesticide" in chem

    @patch("agents.treatment.Groq")
    def test_pest_organic_uses_bt_spray(self, mock_groq):
        mock_groq.return_value = _make_mock_client({"immediate_action": "Remove larvae"})
        result = get_treatment("American Bollworm", 3, "Cotton", problem_type="pest")
        organic = result["organic_treatment"]
        assert "method" in organic
        # Bt (Bacillus thuringiensis) should be in the organic default for pests
        assert "Bt" in organic["method"] or "neem" in organic["method"].lower()
