"""wete.cli.paper_numbers: number formatting and macro names (the paper's numbers come only from results/)."""

import pytest

from wete.cli import paper_numbers as P


def test_format_uses_a_real_minus_and_fixed_decimals():
    assert P.f(-10.55, 1) == "$-$10.6" or P.f(-10.55, 1) == "$-$10.5"
    assert P.f(4.74, 1, signed=True) == "+4.7"
    assert P.ci(-0.126, -0.087, 1, 100.0) == "[$-$12.6, $-$8.7]"


def test_macro_names_are_letters_only():
    m = P.Macros()
    m["TauNavsimBridgesim"] = "$-$0.29"
    assert "\\csname wete@TauNavsimBridgesim\\endcsname{$-$0.29}" in m.tex()
    with pytest.raises(ValueError):
        m["Tau1"] = "x"
