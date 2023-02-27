import numpy as np
import os
import scipy.io

imageID = 0
for root, dirs, files in os.walk("/home/xsw/DLWKB/Dataset/Massachusetts_Roads/all/key_points"):
    for file in files:
        imageID = imageID + 1
        
        key_points_info = scipy.io.loadmat("key_points/" + file)
        if_key_points = key_points_info["if_key_points"]
        all_key_points_position= key_points_info["all_key_points_position"]

        # print(if_key_points.shape)
        # print(all_key_points_position.shape)

        if_key_points = if_key_points.reshape((1,64,64))
        all_key_points_position = all_key_points_position.transpose(1,0)
        all_key_points_position = all_key_points_position.reshape((2,64,64))

        # print(if_key_points.shape)
        # print(all_key_points_position.shape)

        final_mat_savepath = "key_points_final/" + file
        scipy.io.savemat(final_mat_savepath, mdict={'if_key_points': if_key_points, 'all_key_points_position':all_key_points_position})
        print("Image " + str(imageID) + ": Finished!")

# 此py文件用于将关键点信息.mat中的数组的格式进行转换，if_key_points从(1,4096)变为(1,64,64)，all_key_points_position从(4096,2)变为(2,64,64)