import numpy as np
import os
import cv2
import scipy.io

imageID = 0
for root, dirs, files in os.walk("/home/xsw/DLWKB/Dataset/Massachusetts_Roads/all/scribble"):
    for file in files:
        imageID = imageID + 1

        '''
        if file != 'img-14_1_0.png':
            continue
        '''

        image = cv2.imread("scribble/" + file)
        img = image[:,:,0]

        count_up3 = 0 # 所有度大于等于3的道路点的数量，潜在道路交汇点
        count_2 = 0 # 所有度等于2的道路点的数量，一般道路点
        count_1 = 0 # 所有度等于1的道路点的数量，端点

        if_key_points = []
        all_key_points_position = []

        for i in range(0, 64):
            for j in range(0, 64):
                if_exist_road_node = 0 # 是否存在道路点的flag
                count_up3_anchorij = 0 # anchorij内所有度大于等于3的道路点的数量，潜在道路交汇点
                count_2_anchorij = 0 # anchorij内所有度等于2的道路点的数量，一般道路点
                count_1_anchorij = 0 # anchorij内所有度等于1的道路点的数量，端点
                up3_node_list = [] # anchorij内所有度大于等于3的道路点的坐标表
                d2_node_list = [] # anchorij内所有度等于2的道路点的坐标表
                d1_node_list = [] # anchorij内所有度等于1的道路点的坐标表
                for m in range(16*i,16*i+16):
                    for n in range(16*j,16*j+16):
                        if img[m][n] == 1: # 若为道路点
                            if_exist_road_node = 1
                            d = 0 # 坐标为m,n的道路点的度
                            if (m-1>=0) and (m-1<1024) and (n-1>=0) and (n-1<1024):
                                if img[m-1][n-1] == 1:
                                    d = d + 1
                            if (m-1>=0) and (m-1<1024) and (n>=0) and (n<1024):
                                if img[m-1][n] == 1:
                                    d = d + 1
                            if (m-1>=0) and (m-1<1024) and (n+1>=0) and (n+1<1024):
                                if img[m-1][n+1] == 1:
                                    d = d + 1
                            if (m>=0) and (m<1024) and (n-1>=0) and (n-1<1024):
                                if img[m][n-1] == 1:
                                    d = d + 1
                            if (m>=0) and (m<1024) and (n+1>=0) and (n+1<1024):
                                if img[m][n+1] == 1:
                                    d = d + 1
                            if (m+1>=0) and (m+1<1024) and (n-1>=0) and (n-1<1024):
                                if img[m+1][n-1] == 1:
                                    d = d + 1
                            if (m+1>=0) and (m+1<1024) and (n>=0) and (n<1024):
                                if img[m+1][n] == 1:
                                    d = d + 1
                            if (m+1>=0) and (m+1<1024) and (n+1>=0) and (n+1<1024):
                                if img[m+1][n+1] == 1:
                                    d = d + 1
                            if d == 1:
                                count_1 = count_1 + 1
                                count_1_anchorij = count_1_anchorij + 1
                                d1_node_list.append([m,n])
                            elif d == 2:
                                count_2 = count_2 + 1
                                count_2_anchorij = count_2_anchorij + 1
                                d2_node_list.append([m,n])
                            elif d >= 3:
                                count_up3 = count_up3 + 1
                                count_up3_anchorij = count_up3_anchorij + 1
                                up3_node_list.append([m,n])
                if (count_1_anchorij == 0) and (count_2_anchorij == 0) and (count_up3_anchorij == 0): # 不存在道路点 或 存在道路点，但是所有道路点的度均为0(即均为孤立道路点)
                    if_exist_road_node = 0
                key_point = []
                if if_exist_road_node == 0:
                    key_point = [-1, -1]
                elif if_exist_road_node == 1:
                    if_exist_intersection = 0
                    if_exist_endpoint = 0
                    if count_up3_anchorij >= 1:
                        for position in up3_node_list:
                            if ((img[position[0]][position[1]-1]==1) and (img[position[0]][position[1]+1]==1) and (img[position[0]-1][position[1]]==1)) or \
                                ((img[position[0]][position[1]-1]==1) and (img[position[0]][position[1]+1]==1) and (img[position[0]+1][position[1]]==1)) or \
                                ((img[position[0]][position[1]-1]==1) and (img[position[0]-1][position[1]]==1) and (img[position[0]+1][position[1]]==1)) or \
                                ((img[position[0]][position[1]+1]==1) and (img[position[0]-1][position[1]]==1) and (img[position[0]+1][position[1]]==1)) or \
                                ((img[position[0]][position[1]-1]==1) and (img[position[0]-1][position[1]]==1) and (img[position[0]+1][position[1]+1]==1)) or \
                                ((img[position[0]][position[1]+1]==1) and (img[position[0]+1][position[1]]==1) and (img[position[0]-1][position[1]-1]==1)) or \
                                ((img[position[0]][position[1]-1]==1) and (img[position[0]+1][position[1]]==1) and (img[position[0]-1][position[1]+1]==1)) or \
                                ((img[position[0]][position[1]+1]==1) and (img[position[0]-1][position[1]]==1) and (img[position[0]+1][position[1]-1]==1)) or \
                                ((img[position[0]-1][position[1]-1]==1) and (img[position[0]+1][position[1]-1]==1) and (img[position[0]][position[1]+1]==1) and (img[position[0]][position[1]-1]!=1)) or \
                                ((img[position[0]-1][position[1]-1]==1) and (img[position[0]-1][position[1]+1]==1) and (img[position[0]+1][position[1]]==1) and (img[position[0]-1][position[1]]!=1)) or \
                                ((img[position[0]-1][position[1]+1]==1) and (img[position[0]+1][position[1]+1]==1) and (img[position[0]][position[1]-1]==1) and (img[position[0]][position[1]+1]!=1)) or \
                                ((img[position[0]+1][position[1]-1]==1) and (img[position[0]+1][position[1]+1]==1) and (img[position[0]-1][position[1]]==1) and (img[position[0]+1][position[1]]!=1)) or \
                                ((img[position[0]-1][position[1]-1]==1) and (img[position[0]+1][position[1]-1]==1) and (img[position[0]-1][position[1]+1]==1) and (img[position[0]][position[1]-1]!=1) and (img[position[0]-1][position[1]]!=1)) or \
                                ((img[position[0]-1][position[1]-1]==1) and (img[position[0]+1][position[1]-1]==1) and (img[position[0]+1][position[1]+1]==1) and (img[position[0]][position[1]-1]!=1) and (img[position[0]+1][position[1]]!=1)) or \
                                ((img[position[0]-1][position[1]-1]==1) and (img[position[0]-1][position[1]+1]==1) and (img[position[0]+1][position[1]+1]==1) and (img[position[0]][position[1]+1]!=1) and (img[position[0]-1][position[1]]!=1)) or \
                                ((img[position[0]-1][position[1]+1]==1) and (img[position[0]+1][position[1]-1]==1) and (img[position[0]+1][position[1]+1]==1) and (img[position[0]][position[1]+1]!=1) and (img[position[0]+1][position[1]]!=1)):
                                if_exist_intersection = 1
                                key_point = [position[0],  position[1]]
                                break
                    if (count_1_anchorij >= 1) and (if_exist_intersection == 0):
                        if_exist_endpoint = 1
                        key_point = [d1_node_list[0][0], d1_node_list[0][1]]
                    if (count_2_anchorij >= 1) and (if_exist_intersection == 0) and (if_exist_endpoint == 0):
                        sum_x_position = 0
                        sum_y_position = 0 
                        for position in d2_node_list:
                            sum_x_position = sum_x_position + position[0]
                            sum_y_position = sum_y_position + position[1]
                        mean_x_position = sum_x_position // len(d2_node_list)
                        mean_y_position = sum_y_position // len(d2_node_list)
                
                        if img[mean_x_position][mean_y_position] == 1:
                            key_point = [mean_x_position, mean_y_position]
                        else:
                            closest_distance_square = 1000
                            closest_point = []
                            for position in d2_node_list:
                                distance_square = (position[0]-mean_x_position)*(position[0]-mean_x_position)+(position[1]-mean_y_position)*(position[1]-mean_y_position)
                                if distance_square < closest_distance_square:
                                    closest_distance_square = distance_square
                                    closest_point = [position[0],position[1]]
                            key_point = [closest_point[0], closest_point[1]]
                if key_point == []:
                    key_point = [-1, -1]
                    if_exist_road_node = 0
                if_key_points.append(if_exist_road_node)
                all_key_points_position.append(key_point)   
        
        if_key_points_array = np.array(if_key_points)
        all_key_points_position_array = np.array(all_key_points_position)
        
        if (if_key_points_array.shape[0] != 4096) or (all_key_points_position_array.shape[0] != 4096) or (all_key_points_position_array.shape[1] != 2):
            print("Image " + file + " has something wrong!")
            break
        
        mat_savepath = "key_points/" + file[:-4] + ".mat"
        scipy.io.savemat(mat_savepath, mdict={'if_key_points': if_key_points, 'all_key_points_position':all_key_points_position})
        print("Image " + str(imageID) + ": Finished!") 