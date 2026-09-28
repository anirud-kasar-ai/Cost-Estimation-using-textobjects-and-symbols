from pipeline.legend_ocr import is_plausible_legend_description, parse_legend_lines


def test_rejects_color_spec():
    assert not is_plausible_legend_description("GREY = T-800 T-800 T-800")


def test_rejects_see_note_and_decimals():
    assert not is_plausible_legend_description("SEE 27 1000 47.801 1.801")


def test_rejects_garbled_single_token():
    assert not is_plausible_legend_description("STING")


def test_keeps_real_legend_row():
    assert is_plausible_legend_description("CURB DETAIL", detail="DTL C/4")
    assert is_plausible_legend_description("PIPE PENETRATION")
    assert is_plausible_legend_description("NETWORK CAMERA")


def test_parse_drops_junk_keeps_equipment():
    lines = [
        ("GREY = T-800 T-800 T-800", 80.0),
        ("2. STING", 70.0),
        ("7. SEE 27 1000 47.801 1.801", 75.0),
        ("1. CURB DETAIL, DTL C/4", 90.0),
        ("4. PIPE PENETRATION, DTL F/4", 88.0),
    ]
    entries = parse_legend_lines(lines)
    nums = {e.number for e in entries}
    assert nums == {1, 4}
    by = {e.number: e.description for e in entries}
    assert "CURB" in by[1]
    assert "PIPE" in by[4]
