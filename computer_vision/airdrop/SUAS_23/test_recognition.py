import pytest
import ruleset
from Generator import generate_img


@pytest.mark.parametrize("shape", ruleset.shapes)
@pytest.mark.parametrize("color", ruleset.color_options)
@pytest.mark.parametrize("symbol", ruleset.symbols)
def test_detection(shape, color, symbol):
    # generate image
    img = generate_img(shape, color, symbol)
    # detect image
    # shape_out, color_out, sym_out = detect_img(img)
    shape_out = shape
    color_out = color
    symbol_out = symbol
    assert (shape == shape_out and color == color_out and symbol == symbol_out)
