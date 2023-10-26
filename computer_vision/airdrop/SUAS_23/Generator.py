

import random
from PIL import Image, ImageDraw, ImageFont
from ruleset import shapes, color_options, symbols

# Random generator for Shape, Color and Symbol


def save_random_color_image(filename, img_h, img_w):
    selected_shape = random.choice(shapes)
    selected_color = random.choice(color_options)
    while True:
        color2 = random.choice(color_options)
        if color2 != selected_color:
            break

    selected_symbol = random.choice(symbols)
    # selected_shape = "Quarter_circle"

    print(selected_shape)
    print(selected_color)

    gray = (128, 128, 128)

    # ('RGB', (100, 100), selected_color)
    img = Image.new('RGB', (img_h, img_w), gray)

    draw = ImageDraw.Draw(img)
    center = (0.5*img_w, 0.5*img_h)
    radius = 0.4*img_w

    if selected_shape == "Circle":
        draw.ellipse([center[0]-radius, center[1]-radius,
                     center[0]+radius, center[1]+radius], fill=selected_color)
    elif selected_shape == "Semi-Circle":
        center = (0.5*img_w, 0.35*img_h)
        draw.pieslice([center[0]-radius, center[1]-radius, center[0] +
                      radius, center[1]+radius], 0, 180, fill=selected_color)
    elif selected_shape == "Quarter_circle":
        center = (0.325*img_w, 0.35*img_h)
        draw.pieslice([center[0]-radius, center[1]-radius, center[0] +
                      radius, center[1]+radius], 0, 90, fill=selected_color)
    elif selected_shape == "Star":
        # Define points for a star
        points = [
            (0.5*img_w, 0.1*img_h),
            (0.61*img_w, 0.5*img_h),
            (1.0*img_w, 0.5*img_h),
            (0.7*img_w, 0.7*img_h),
            (0.8*img_w, 1.0*img_h),
            (0.5*img_w, 0.8*img_h),
            (0.2*img_w, 1.0*img_h),
            (0.3*img_w, 0.7*img_h),
            (0, 0.5*img_h),
            (0.39*img_w, 0.5*img_h)
        ]
        draw.polygon(points, fill=selected_color)
    elif selected_shape == "Pentagon":
        # Define points for a regular pentagon
        points = [
            (.50 * img_w, img_h * .05),
            (img_w * 0.95, img_h * .45),
            (.80 * img_w, img_h * 0.95),
            (.20 * img_w, img_h * 0.95),
            (img_w * .05, img_h * .45),
        ]
        draw.polygon(points, fill=selected_color)
    elif selected_shape == "Triangle":
        # Define points for an equilateral triangle
        triangle = [
            (img_w * .50, img_h * .10),
            (img_w * .90, img_h * .90),
            (img_w * .10, img_h * .90),
        ]
        draw.polygon(triangle, fill=selected_color)
    elif selected_shape == "Rectangle":
        # Define points for a rectangle
        draw.rectangle([.2 * img_w, .2 * img_h, .80 *
                       img_w, .80 * img_h], fill=selected_color)
    elif selected_shape == "Cross":
        # Define points for a cross
        draw.rectangle([img_w*.20, img_h*.40, img_w*.80,
                       img_h*.60], fill=selected_color)
        draw.rectangle([img_w*.40, img_h*.20, img_w*.60,
                       img_h*.80], fill=selected_color)

    font = ImageFont.truetype("Roboto/Roboto-Black.ttf", size=50)
    draw.text((img_h * 0.465, img_w * 0.465),
              selected_symbol, color2, font=font)

    img.save(filename)


def generate_img(shape, color, symbol):
    img_h = 500
    img_w = 500
    selected_shape = shape
    selected_color = color
    while True:
        color2 = random.choice(color_options)
        if color2 != selected_color:
            break

    selected_symbol = symbol
    # selected_shape = "Quarter_circle"

    print(selected_shape)
    print(selected_color)

    gray = (128, 128, 128)

    # ('RGB', (100, 100), selected_color)
    img = Image.new('RGB', (img_h, img_w), gray)

    draw = ImageDraw.Draw(img)
    center = (0.5*img_w, 0.5*img_h)
    radius = 0.4*img_w

    if selected_shape == "Circle":
        draw.ellipse([center[0]-radius, center[1]-radius,
                     center[0]+radius, center[1]+radius], fill=selected_color)
    elif selected_shape == "Semi-Circle":
        center = (0.5*img_w, 0.35*img_h)
        draw.pieslice([center[0]-radius, center[1]-radius, center[0] +
                      radius, center[1]+radius], 0, 180, fill=selected_color)
    elif selected_shape == "Quarter_circle":
        center = (0.325*img_w, 0.35*img_h)
        draw.pieslice([center[0]-radius, center[1]-radius, center[0] +
                      radius, center[1]+radius], 0, 90, fill=selected_color)
    elif selected_shape == "Star":
        # Define points for a star
        points = [
            (0.5*img_w, 0.1*img_h),
            (0.61*img_w, 0.5*img_h),
            (1.0*img_w, 0.5*img_h),
            (0.7*img_w, 0.7*img_h),
            (0.8*img_w, 1.0*img_h),
            (0.5*img_w, 0.8*img_h),
            (0.2*img_w, 1.0*img_h),
            (0.3*img_w, 0.7*img_h),
            (0, 0.5*img_h),
            (0.39*img_w, 0.5*img_h)
        ]
        draw.polygon(points, fill=selected_color)
    elif selected_shape == "Pentagon":
        # Define points for a regular pentagon
        points = [
            (.50 * img_w, img_h * .05),
            (img_w * 0.95, img_h * .45),
            (.80 * img_w, img_h * 0.95),
            (.20 * img_w, img_h * 0.95),
            (img_w * .05, img_h * .45),
        ]
        draw.polygon(points, fill=selected_color)
    elif selected_shape == "Triangle":
        # Define points for an equilateral triangle
        triangle = [
            (img_w * .50, img_h * .10),
            (img_w * .90, img_h * .90),
            (img_w * .10, img_h * .90),
        ]
        draw.polygon(triangle, fill=selected_color)
    elif selected_shape == "Rectangle":
        # Define points for a rectangle
        draw.rectangle([.2 * img_w, .2 * img_h, .80 *
                       img_w, .80 * img_h], fill=selected_color)
    elif selected_shape == "Cross":
        # Define points for a cross
        draw.rectangle([img_w*.20, img_h*.40, img_w*.80,
                       img_h*.60], fill=selected_color)
        draw.rectangle([img_w*.40, img_h*.20, img_w*.60,
                       img_h*.80], fill=selected_color)

    font = ImageFont.truetype("Roboto/Roboto-Black.ttf", size=50)
    draw.text((img_h * 0.465, img_w * 0.465),
              selected_symbol, color2, font=font)

    return img


if __name__ == "__main__":
    save_random_color_image('test.png', 500, 500)
