import numpy as np
import cv2
import scipy.io
import os

def search_link(img, search_point_list, key_points_gt_list, point_searched_list, key_point_searched_list, i, j, anchor_link, iter_count):
    iter_count = iter_count + 1
    if search_point_list == [] or iter_count >= 100:
        return "Search Finished"
    else:
        next_search_list = []
        for search_point in search_point_list:
            if search_point not in point_searched_list:
                point_searched_list.append(search_point)
                normal_point_list = []
                key_points_count = 0
                if (search_point[0]-1>=0) and (search_point[0]-1<1024) and (search_point[1]>=0) and (search_point[1]<1024):
                    if img[search_point[0]-1][search_point[1]]==1:
                        if [search_point[0]-1,search_point[1]] in key_points_gt_list:
                            key_points_count = key_points_count + 1
                            if [search_point[0]-1,search_point[1]] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]-1,search_point[1]])
                                anchor_link[key_points_gt_list.index([search_point[0]-1,search_point[1]]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]-1,search_point[1]])
                if (search_point[0]-1>=0) and (search_point[0]-1<1024) and (search_point[1]+1>=0) and (search_point[1]+1<1024):
                    if img[search_point[0]-1][search_point[1]+1]==1:
                        if [search_point[0]-1,search_point[1]+1] in key_points_gt_list:
                            key_points_count = key_points_count + 1
                            if [search_point[0]-1,search_point[1]+1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]-1,search_point[1]+1])
                                anchor_link[key_points_gt_list.index([search_point[0]-1,search_point[1]+1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]-1,search_point[1]+1])
                if (search_point[0]>=0) and (search_point[0]<1024) and (search_point[1]+1>=0) and (search_point[1]+1<1024):
                    if img[search_point[0]][search_point[1]+1]==1:
                        if [search_point[0],search_point[1]+1] in key_points_gt_list:
                            key_points_count = key_points_count + 1
                            if [search_point[0],search_point[1]+1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0],search_point[1]+1])
                                anchor_link[key_points_gt_list.index([search_point[0],search_point[1]+1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0],search_point[1]+1])
                if (search_point[0]+1>=0) and (search_point[0]+1<1024) and (search_point[1]+1>=0) and (search_point[1]+1<1024):
                    if img[search_point[0]+1][search_point[1]+1]==1:
                        if [search_point[0]+1,search_point[1]+1] in key_points_gt_list:
                            key_points_count = key_points_count + 1
                            if [search_point[0]+1,search_point[1]+1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]+1,search_point[1]+1])
                                anchor_link[key_points_gt_list.index([search_point[0]+1,search_point[1]+1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]+1,search_point[1]+1])
                if (search_point[0]+1>=0) and (search_point[0]+1<1024) and (search_point[1]>=0) and (search_point[1]<1024):
                    if img[search_point[0]+1][search_point[1]]==1:
                        if [search_point[0]+1,search_point[1]] in key_points_gt_list:
                            key_points_count = key_points_count + 1
                            if [search_point[0]+1,search_point[1]] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]+1,search_point[1]])
                                anchor_link[key_points_gt_list.index([search_point[0]+1,search_point[1]]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]+1,search_point[1]])
                if (search_point[0]+1>=0) and (search_point[0]+1<1024) and (search_point[1]-1>=0) and (search_point[1]-1<1024):
                    if img[search_point[0]+1][search_point[1]-1]==1:
                        if [search_point[0]+1,search_point[1]-1] in key_points_gt_list:
                            key_points_count = key_points_count + 1
                            if [search_point[0]+1,search_point[1]-1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]+1,search_point[1]-1])
                                anchor_link[key_points_gt_list.index([search_point[0]+1,search_point[1]-1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]+1,search_point[1]-1])
                if (search_point[0]>=0) and (search_point[0]<1024) and (search_point[1]-1>=0) and (search_point[1]-1<1024):
                    if img[search_point[0]][search_point[1]-1]==1:
                        if [search_point[0],search_point[1]-1] in key_points_gt_list:
                            key_points_count = key_points_count + 1
                            if [search_point[0],search_point[1]-1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0],search_point[1]-1])
                                anchor_link[key_points_gt_list.index([search_point[0],search_point[1]-1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0],search_point[1]-1])
                if (search_point[0]-1>=0) and (search_point[0]-1<1024) and (search_point[1]-1>=0) and (search_point[1]-1<1024):
                    if img[search_point[0]-1][search_point[1]-1]==1:
                        if [search_point[0]-1,search_point[1]-1] in key_points_gt_list:
                            key_points_count = key_points_count + 1
                            if [search_point[0]-1,search_point[1]-1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]-1,search_point[1]-1])
                                anchor_link[key_points_gt_list.index([search_point[0]-1,search_point[1]-1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]-1,search_point[1]-1])
                if key_points_count == 0:
                    for normal_point in normal_point_list:
                        if normal_point not in next_search_list:
                            next_search_list.append(normal_point)
        return search_link(img, next_search_list, key_points_gt_list, point_searched_list, key_point_searched_list, i, j, anchor_link, iter_count)

imageID = 0
for root, dirs, files in os.walk("/home/xsw/DLWKB/Dataset/Massachusetts_Roads/all/key_points_final"):
    for file in files:
        imageID = imageID + 1
        key_points_info = scipy.io.loadmat("key_points_final/" + file)
        if_key_points = key_points_info["if_key_points"]
        all_key_points_position= key_points_info["all_key_points_position"]

        mask = cv2.imread("scribble/" + file[:-4] + ".png")
        img = mask[:,:,0]
        anchor_link = np.zeros((8,64,64))

        for i in range(0, 64):
            for j in range(0, 64):
                if if_key_points[0,i,j] == 0:
                    for k in range(0,8):
                        anchor_link[k,i,j] = -1
                elif if_key_points[0,i,j] == 1:
                    keypoints = []
                    keypoints_surrounds = []
                    keypoints.append([all_key_points_position[0,i,j],all_key_points_position[1,i,j]])
                    if (i-1>=0) and (i-1<64) and (j>=0) and (j<64):
                        keypoints_surrounds.append([all_key_points_position[0,i-1,j],all_key_points_position[1,i-1,j]])
                    else:
                        keypoints_surrounds.append([-1,-1])
                    if (i-1>=0) and (i-1<64) and (j+1>=0) and (j+1<64):
                        keypoints_surrounds.append([all_key_points_position[0,i-1,j+1],all_key_points_position[1,i-1,j+1]])
                    else:
                        keypoints_surrounds.append([-1,-1])
                    if (i>=0) and (i<64) and (j+1>=0) and (j+1<64):
                        keypoints_surrounds.append([all_key_points_position[0,i,j+1],all_key_points_position[1,i,j+1]])
                    else:
                        keypoints_surrounds.append([-1,-1])
                    if (i+1>=0) and (i+1<64) and (j+1>=0) and (j+1<64):
                        keypoints_surrounds.append([all_key_points_position[0,i+1,j+1],all_key_points_position[1,i+1,j+1]])
                    else:
                        keypoints_surrounds.append([-1,-1])
                    if (i+1>=0) and (i+1<64) and (j>=0) and (j<64):
                        keypoints_surrounds.append([all_key_points_position[0,i+1,j],all_key_points_position[1,i+1,j]])
                    else:
                        keypoints_surrounds.append([-1,-1])
                    if (i+1>=0) and (i+1<64) and (j-1>=0) and (j-1<64):
                        keypoints_surrounds.append([all_key_points_position[0,i+1,j-1],all_key_points_position[1,i+1,j-1]])
                    else:
                        keypoints_surrounds.append([-1,-1])
                    if (i>=0) and (i<64) and (j-1>=0) and (j-1<64):
                        keypoints_surrounds.append([all_key_points_position[0,i,j-1],all_key_points_position[1,i,j-1]])
                    else:
                        keypoints_surrounds.append([-1,-1])
                    if (i-1>=0) and (i-1<64) and (j-1>=0) and (j-1<64):
                        keypoints_surrounds.append([all_key_points_position[0,i-1,j-1],all_key_points_position[1,i-1,j-1]])
                    else:
                        keypoints_surrounds.append([-1,-1])

                    point_searched_list = []
                    key_point_searched_list = []
                    iter_count = 0
                    search_link(img, keypoints, keypoints_surrounds, point_searched_list, key_point_searched_list, i, j, anchor_link, iter_count)
        
        final_mat_savepath = "link_key_points_final/" + file
        scipy.io.savemat(final_mat_savepath, mdict={'if_key_points': if_key_points, 'all_key_points_position':all_key_points_position, 'anchor_link':anchor_link})
        print("Image " + str(imageID) + ": Finished!")
