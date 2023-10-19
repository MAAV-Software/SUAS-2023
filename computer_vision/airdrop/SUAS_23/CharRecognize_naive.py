import cv2
import pytesseract
import matplotlib.pyplot as plt


def processImg(img):
    '''Perform any processing.

    grayscale, threshold, dilation
    '''
    img1 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)

    th = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if cv2.contourArea(contour) > 170:
            [X, Y, W, H] = cv2.boundingRect(contour)
            cv2.rectangle(img1, (X, Y), (X + W, Y + H), (0, 0, 255), 2)
            Cropped0 = th[Y - 2:Y + H + 2, X - 2:X + W + 2]
            return Cropped0  # STACK OVERFLOW https://stackoverflow.com/questions/72030362/how-to-perform-ocr-for-several-contours
    return edged


def parseImg(img):
    hImg, wImg = img.shape[:2]
    print(f"Image dimensions: Height={hImg}, Width={wImg}")
    # crop image

    # Set the size of the cropped region
    crop_size = 115

    # Calculate the crop coordinates
    start_x = wImg // 2 - crop_size // 2
    start_y = hImg // 2 - crop_size // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    # Crop the image to the center
    cropped_img = img[start_y:end_y, start_x:end_x]
    cropped_img = cv2.resize(cropped_img, (500, 500))
    text = pytesseract.image_to_string(cropped_img, config=(
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 10"))
    if text:
        print("Detected text:")
        print(text)
    else:
        print("No text detected.")

    boxes = pytesseract.image_to_boxes(
        cropped_img, config=("-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 10"))

    if boxes:
        for b in boxes.splitlines():
            b = b.split(' ')
            print(b)
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(cropped_img, (x, 500 - y),
                          (w, hImg - h), (0, 0, 0), 1)
            cv2.putText(cropped_img, b[0], (x, 500 - y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)



# Load image
# test 1 no change
image = cv2.imread('test.png')
parseImg(image)
# test 2 gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
parseImg(gray)
# test 3 with preprocessing
processedImg = processImg(image)
parseImg(processedImg)


# Display or use the results as needed
# cv2.imshow('Processed IMmage', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()