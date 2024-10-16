import pandas as pd
import numpy as np
import cv2

def load_csv(file_path):
    return pd.read_csv(file_path)

def transform_coordinates(csv_data):
    longitudes = csv_data['Longitude'].values
    latitudes = csv_data['Latitude'].values
    return longitudes, latitudes

def resize_to_same_height(img1, img2):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    if h1 > h2:
        img1 = cv2.resize(img1, (w1, h2))
    else:
        img2 = cv2.resize(img2, (w2, h1))
    return img1, img2

def stitch_images(image_paths, csv_files):
    stitched_image = None
    stitched_longitudes = []
    stitched_latitudes = []
    
    for i in range(len(image_paths)):
        image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)  
        csv_data = load_csv(csv_files[i])

        longitudes, latitudes = transform_coordinates(csv_data)
        
        if stitched_image is None:
            stitched_image = image
            stitched_longitudes.extend(longitudes)  
            stitched_latitudes.extend(latitudes)  
        else:
            previous_csv_data = load_csv(csv_files[i-1])
            prev_longitudes, prev_latitudes = transform_coordinates(previous_csv_data)
            
            overlap_indices = np.intersect1d(prev_longitudes, longitudes, return_indices=True)
            
            if overlap_indices[0].size == 0:
                print("No overlap found for image", i)
                continue
            
            overlap_prev_idx = overlap_indices[1]
            overlap_cur_idx = overlap_indices[2]
            
            stitched_image, image = resize_to_same_height(stitched_image, image)

            overlap_start_idx = overlap_cur_idx[0]  
            
            stitched_image = np.hstack((stitched_image, image[:, overlap_start_idx:]))
            
            stitched_longitudes.extend(longitudes[overlap_start_idx:])  
            stitched_latitudes.extend(latitudes[overlap_start_idx:])  
            


    return stitched_image, stitched_longitudes, stitched_latitudes

image_files = ['pic_1.png', 'pic_2.png', 'pic_3.png']
csv_files = ['csv_1.csv', 'csv_2.csv', 'csv_3.csv']

stitched_result, longitudes, latitudes = stitch_images(image_files, csv_files)

save_success = cv2.imwrite('stitched_image.png', stitched_result)
if save_success:
    print("Image saved successfully!")
else:
    print("Error saving image.")

stitched_csv_data = pd.DataFrame({'Longitude': longitudes, 'Latitude': latitudes})
stitched_csv_data.to_csv('stitched_coordinates.csv', index=False)
print("Stitched coordinates saved to stitched_coordinates.csv.")

cv2.imshow('Stitched Image', stitched_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
