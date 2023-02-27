import numpy as np
import cv2
import scipy.io

key_points_info = scipy.io.loadmat("key_points_final/img-14_1_0.mat") # 
if_key_points= key_points_info["if_key_points"]
all_key_points_position= key_points_info["all_key_points_position"]

file = "img-14_1_0.png" # 
image = cv2.imread("scribble/" + file)
img = image[:,:,0]
new_img = np.zeros((1024,1024,3))
for i in range(0, 64):
    for j in range(0, 64):
        for m in range(16*i,16*i+16):
            for n in range(16*j,16*j+16):
                new_img[m][n] = [255,255,255]
                if if_key_points[0,i,j] == 0:
                    new_img[m][n] = [0,255,255]
                if img[m][n] == 1:
                    new_img[m][n] = [0,0,0]
                if (all_key_points_position[0,i,j]==m) and (all_key_points_position[1,i,j]==n):
                    new_img[m][n] = [0,0,255]
cv2.imwrite("img-14_1_0-test.png", new_img)
print("New image generating finished!")

# 此py文件验证生成的mat里包含的关键点信息是否与检测到的一致