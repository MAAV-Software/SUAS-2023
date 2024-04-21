# import cv2 
# from stitching import Stitcher
# from imgstitch import stitch_images_and_save
# # image_paths=['test_1_.jpg','test_2_.jpg','test_3_.jpg'] 
# # # initialized a list of images 
# # imgs = [] 
  
# # settings = {"detector": "sift", "confidence_threshold": 0.2}
# # stitcher = Stitcher(**settings)

# # panorama = stitcher.stitch(["test_1_.jpg", "test_2_.jpg", "test_3_.jpg", "test_4_.jpg", "test_5_.jpg"])

# stitch_images_and_save("", ["test_1_.jpg", "test_2_.jpg", "test_3_.jpg", "test_4_.jpg", "test_5_.jpg"], 1, output_folder="")
# # cv2.imwrite('panorama.png', panorama)
# # for i in range(len(image_paths)): 
# #     imgs.append(cv2.imread(image_paths[i])) 
# #     imgs[i]=cv2.resize(imgs[i],(0,0),fx=0.4,fy=0.4) 
# #     # this is optional if your input images isn't too large 
# #     # you don't need to scale down the image 
# #     # in my case the input images are of dimensions 3000x1200 
# #     # and due to this the resultant image won't fit the screen 
# #     # scaling down the images  
# # # showing the original pictures 
# # # cv2.imshow('1',imgs[0]) 
# # # cv2.imshow('2',imgs[1]) 
# # # cv2.imshow('3',imgs[2]) 
  
# # stitchy=cv2.Stitcher.create() 
# # (dummy,output)=stitchy.stitch(imgs) 
  
# # if dummy != cv2.STITCHER_OK: 
# #     print(dummy)
# #   # checking if the stitching procedure is successful 
# #   # .stitch() function returns a true value if stitching is  
# #   # done successfully 
# #     print("stitching ain't successful") 
# # else:  
# #     print('Your Panorama is ready!!!') 
  
# # # final output
# # cv2.imwrite("stitched.png", output)   
# # cv2.waitKey(0)


import cv2
import numpy as np

# Function to stitch images
def stitch_images(images):
    # Initialize the stitcher
    stitcher = cv2.Stitcher_create() if cv2.__version__.startswith('4') else cv2.Stitcher_create()
    # Attempt to stitch the images
    # stitcher.setPanoConfidenceThresh()
    status, stitched = stitcher.stitch(images)
    # If successful, return the stitched image
    if status == cv2.Stitcher_OK:
        return stitched
    else:
        print("Stitching failed!")
        print(status)
        return None

# Load your similar images
image1 = cv2.imread('test_1_.jpg')
image2 = cv2.imread('test_2_.jpg')
image3 = cv2.imread('test_3_.jpg')
image4 = cv2.imread('test_4_.jpg')
# image5 = cv2.imread('test_5_.jpg')


# Resize the images to the same dimensions (optional but recommended)
width = 800
height = 600
image1 = cv2.resize(image1, (width, height))
image2 = cv2.resize(image2, (width, height))
image3 = cv2.resize(image3, (width, height))
image4 = cv2.resize(image4, (width, height))
# image5 = cv2.resize(image5, (width, height))


# Call the stitch_images function with your images
result_image = stitch_images([image1, image2, image3, image4])

# Display the result
if result_image is not None:
    cv2.imshow('Stitched Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
