import os
import sys
import unittest
from unittest.mock import patch
from io import StringIO
import tempfile
import sankhya

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import assistant

VALID_RESPONSE = """```lp
Maximize
 obj: x + y
Subject To
 c1: x <= 1
 c2: y <= 1
End
```
```mps
NAME          TEST
OBJSENSE
 MAXIMIZE
ROWS
 N  obj
 L  c1
 L  c2
COLUMNS
    x         obj       1.0
    x         c1        1.0
    y         obj       1.0
    y         c2        1.0
RHS
    RHS1      c1        1.0
    RHS1      c2        1.0
BOUNDS
ENDATA
```
"""

VALID_RESPONSE_SCI = """```lp
Maximize
 obj: 1e0 x + 10e-1 y
Subject To
 c1: x <= 1.000000000
 c2: y <= +1
End
```
```mps
NAME          TEST
OBJSENSE
 MAXIMIZE
ROWS
 N  obj
 L  c1
 L  c2
COLUMNS
    x         obj       1.0
    x         c1        1.0
    y         obj       1.0
    y         c2        1.0
RHS
    RHS1      c1        1.0
    RHS1      c2        1.0
BOUNDS
ENDATA
```
"""

SWAPPED_COEFFS_LP = """```lp
Maximize
 obj: y + x
Subject To
 c1: y <= 1
 c2: x <= 1
End
```"""

DIFF_VAR_BOUNDS_LP = """```lp
Maximize
 obj: x + y
Subject To
 c1: x <= 1
 c2: y <= 1
Bounds
 0 <= x <= 2
End
```"""

DIFF_CONS_BOUNDS_LP = """```lp
Maximize
 obj: x + y
Subject To
 c1: x <= 2
 c2: y <= 1
End
```"""

CONTINUOUS_VS_INTEGER_LP = """```lp
Maximize
 obj: x + y
Subject To
 c1: x <= 1
 c2: y <= 1
Integer
 x
End
```"""

DIFF_OBJ_LP = """```lp
Minimize
 obj: x + y
Subject To
 c1: x <= 1
 c2: y <= 1
End
```"""

class TestAssistant(unittest.TestCase):
    def setUp(self):
        self.held, sys.stdout = sys.stdout, StringIO()

    def tearDown(self):
        sys.stdout = self.held

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": ""}, clear=True)
    def test_no_endpoint_configured(self):
        result = assistant.ask_assistant("Test")
        self.assertIsNone(result)

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": "http://stub"}, clear=True)
    @patch('assistant.generate_lp')
    @patch('builtins.input')
    def test_successful_generation(self, mock_input, mock_generate):
        mock_input.return_value = ""
        mock_generate.return_value = VALID_RESPONSE
        result = assistant.ask_assistant("Test")
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "optimal")

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": "http://stub"}, clear=True)
    @patch('assistant.generate_lp')
    @patch('builtins.input')
    def test_scientific_notation_equivalent(self, mock_input, mock_generate):
        mock_input.return_value = ""
        mock_generate.return_value = VALID_RESPONSE_SCI
        result = assistant.ask_assistant("Test")
        self.assertIsNotNone(result)

    def _test_rejection(self, invalid_lp, mock_correct, mock_generate):
        mps_part = VALID_RESPONSE.split("```mps")[1]
        mock_generate.return_value = invalid_lp + "\n```mps" + mps_part
        mock_correct.return_value = "STILL INVALID"
        result = assistant.ask_assistant("Test")
        self.assertIsNone(result)
        self.assertEqual(mock_correct.call_count, 3)

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": "http://stub"}, clear=True)
    @patch('assistant.generate_lp')
    @patch('assistant.correct_lp')
    def test_swapped_coefficients_rejected(self, mock_correct, mock_generate):
        self._test_rejection(SWAPPED_COEFFS_LP, mock_correct, mock_generate)

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": "http://stub"}, clear=True)
    @patch('assistant.generate_lp')
    @patch('assistant.correct_lp')
    def test_different_variable_bounds_rejected(self, mock_correct, mock_generate):
        self._test_rejection(DIFF_VAR_BOUNDS_LP, mock_correct, mock_generate)

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": "http://stub"}, clear=True)
    @patch('assistant.generate_lp')
    @patch('assistant.correct_lp')
    def test_different_constraint_bounds_rejected(self, mock_correct, mock_generate):
        self._test_rejection(DIFF_CONS_BOUNDS_LP, mock_correct, mock_generate)

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": "http://stub"}, clear=True)
    @patch('assistant.generate_lp')
    @patch('assistant.correct_lp')
    def test_continuous_vs_integer_rejected(self, mock_correct, mock_generate):
        self._test_rejection(CONTINUOUS_VS_INTEGER_LP, mock_correct, mock_generate)

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": "http://stub"}, clear=True)
    @patch('assistant.generate_lp')
    @patch('assistant.correct_lp')
    def test_different_objective_rejected(self, mock_correct, mock_generate):
        self._test_rejection(DIFF_OBJ_LP, mock_correct, mock_generate)

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": "http://stub"}, clear=True)
    @patch('assistant.generate_lp')
    @patch('builtins.input')
    def test_cancellation(self, mock_input, mock_generate):
        mock_input.return_value = "n"
        mock_generate.return_value = VALID_RESPONSE
        result = assistant.ask_assistant("Test")
        self.assertIsNone(result)

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": "http://stub"}, clear=True)
    @patch('assistant.generate_lp')
    @patch('assistant.correct_lp')
    @patch('builtins.input')
    def test_mismatch_triggers_correction_attempt(self, mock_input, mock_correct, mock_generate):
        mps_part = VALID_RESPONSE.split("```mps")[1]
        mock_input.return_value = ""
        mock_generate.return_value = SWAPPED_COEFFS_LP + "\n```mps" + mps_part
        mock_correct.return_value = VALID_RESPONSE # Fixes it on first correction
        result = assistant.ask_assistant("Test")
        self.assertIsNotNone(result)
        self.assertEqual(mock_correct.call_count, 1)

    @patch.dict(os.environ, {"SANKHYA_ASSISTANT_URL": "http://stub"}, clear=True)
    @patch('assistant.generate_lp')
    @patch('assistant.correct_lp')
    def test_persistent_mismatch_stops(self, mock_correct, mock_generate):
        mps_part = VALID_RESPONSE.split("```mps")[1]
        mock_generate.return_value = SWAPPED_COEFFS_LP + "\n```mps" + mps_part
        mock_correct.return_value = SWAPPED_COEFFS_LP + "\n```mps" + mps_part
        result = assistant.ask_assistant("Test")
        self.assertIsNone(result)
        self.assertEqual(mock_correct.call_count, 3)

if __name__ == '__main__':
    unittest.main()
