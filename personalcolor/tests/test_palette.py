import pytest

from personalcolor.palette import PALETTES, SEASONS, get_palette


def test_all_seasons_have_a_palette():
    for season in SEASONS:
        profile = get_palette(season)
        assert profile.name == season
        assert len(profile.swatches) > 0


def test_unknown_season_raises():
    with pytest.raises(ValueError):
        get_palette("Monsoon")


def test_swatches_are_hex_codes():
    for profile in PALETTES.values():
        for hex_code in profile.swatches:
            assert hex_code.startswith("#")
            assert len(hex_code) == 7
