import pytest
import ruleset
from Generator import generate_img


@pytest.mark.parametrize("color", ruleset.colors_dict)
@pytest.mark.parametrize("symbol", ruleset.symbols_dict)
@pytest.mark.parametrize("shape", ruleset.shapes)
def test_detection(color, symbol, shape):
    # generate image
    img = generate_img(color, symbol, shape)
    # detect image
    # shape_out, color_out, sym_out = detect_img(img)
    color_out = color
    symbol_out = symbol
    shape_out = shape
    assert (color == color_out and symbol == symbol_out and shape == shape_out)
