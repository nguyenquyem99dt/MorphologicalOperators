import numpy as np

def erode(img, kernel):
    img_shape = img.shape
    kernel_shape = kernel.shape
    kernel_center = (kernel_shape[0] // 2, kernel_shape[1] // 2)
    eroded_img = np.zeros((img_shape[0] + kernel_shape[0] - 1, img_shape[1] + kernel_shape[1] - 1))
    
    x_append = np.zeros((img.shape[0], kernel_shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel_shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            x,y = i + kernel_center[0], j + kernel_center[1]
            tmp = np.zeros((kernel_shape[0],kernel_shape[1]))
            if np.all(img[i:i_, j:j_]!=0):
                tmp = img[i:i_,j:j_] - kernel[0:kernel.shape[0], 0:kernel.shape[1]]
                eroded_img[x,y] = tmp.min()
    
    return eroded_img[:img_shape[0], :img_shape[1]]/255.0

def dilate(img, kernel):
    kernel_shape = kernel.shape
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    dilated_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            tmp = np.zeros((kernel_shape[0],kernel_shape[1]))
            if (img[i+kernel_center[0],j+kernel_center[0]]!=0):
                tmp = img[i:i_,j:j_] + kernel[0:kernel.shape[0], 0:kernel.shape[1]]
                for m in range(i,i_):
                    for n in range(j,j_):
                        new_value = img[i+kernel_center[0],j+kernel_center[0]]+kernel[kernel_center[0], kernel_center[1]]
                        if(dilated_img[m, n] < new_value):
                            dilated_img[m, n] = new_value
                dilated_img[i+kernel_center[0],j+kernel_center[1]] = tmp.max()
    return dilated_img[:img_shape[0], :img_shape[1]]/255.0

def open(img,kernel):
    return dilate(erode(img,kernel)*255.0, kernel)

def close(img,kernel):
    return erode(dilate(img,kernel)*255.0, kernel)

def smooth(img, kernel):
    return close(open(img,kernel)*255.0, kernel)

def gradient(img, kernel):
    return dilate(img,kernel) - erode(img,kernel)

def top_hat(img,kernel):
    return (img - open(img,kernel)*255.0)/255.0

def bottom_hat(img,kernel):
    return (close(img,kernel)*255.0 - img)/255.0

def granulometry(img, kernel):
    return open(smooth(img,kernel)*255,kernel)

def point_wise_minimum(A,B):
    A_shape = A.shape
    B_shape = B.shape
    if(A_shape[0]!=B_shape[0] or A_shape[1]!=B_shape[1]):
        print('A and B shape must be same!')
        return 0
    result = np.zeros((A_shape[0], A_shape[1]))
    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            result[i,j] = A[i,j] if(A[i,j]<=B[i,j]) else B[i,j]
    return result

def point_wise_maximum(A,B):
    A_shape = A.shape
    B_shape = B.shape
    if(A_shape[0]!=B_shape[0] or A_shape[1]!=B_shape[1]):
        print('A and B shape must be same!')
        return 0
    result = np.zeros((A_shape[0], A_shape[1]))
    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            result[i,j] = A[i,j] if(A[i,j]>=B[i,j]) else B[i,j]
    return result

def reconstruct_by_dilation(marker_img,kernel, mask_img):
    tmp_before = marker_img
    while True:
        tmp_after = point_wise_minimum(255*dilate(tmp_before,kernel),mask_img)
        if(np.all(tmp_after==tmp_before)):
            return tmp_before/255.0
        tmp_before = tmp_after
    return tmp_before/255.0

def reconstruct_by_erosion(marker_img, kernel, mask_img):
    tmp_before = marker_img
    while True:
        tmp_after = point_wise_maximum(255*erode(tmp_before, kernel), mask_img)
        if(np.all(tmp_after==tmp_before)):
            return tmp_before/255.0
        tmp_before = tmp_after
    return tmp_before/255.0

def open_by_reconstruction(marker_img, kernel, mask_img, iteration):
    tmp_before = marker_img
    for i in range(iteration):
        tmp_after = erode(tmp_before,kernel)*255
        tmp_before = tmp_after
    result = reconstruct_by_dilation(tmp_before,kernel, mask_img)
    return result

def close_by_reconstruction(marker_img, kernel, mask_img, iteration):
    tmp_before = marker_img
    for i in range(iteration):
        tmp_after = dilate(tmp_before, kernel)*255
        tmp_before = tmp_after
    result = reconstruct_by_erosion(tmp_before, kernel, mask_img)
    return result
    