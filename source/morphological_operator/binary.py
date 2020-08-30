import numpy as np
import cv2
def dilate(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
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
            if img[i+kernel_center[0], j+kernel_center[1]]==255:
                dilated_img[i:i_, j:j_] = 255
    return dilated_img[:img_shape[0], :img_shape[1]]

def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 255

    return eroded_img[:img_shape[0], :img_shape[1]]

def open(img, kernel):
    return dilate(erode(img,kernel), kernel)

def close(img, kernel):
    return erode(dilate(img,kernel), kernel)

def bitwise_and(X, Y):
    return np.bitwise_and(np.uint8(X),np.uint8(Y))

def hitmiss(img, kernel):
    kernel_hit = np.zeros((kernel.shape[0], kernel.shape[1]))
    kernel_hit[kernel==1] = 1

    kernel_miss = np.zeros((kernel.shape[0], kernel.shape[1]))
    kernel_miss[kernel==-1]=1

    e1 = erode(img,kernel_hit)
    e2 = erode(255-img,kernel_miss)
    return bitwise_and(e1,e2)

def extract_boundary(img, kernel):
    return (img-erode(img, kernel))*255.0 #Khi trừ nó lại lấy nhị phân 1, 0 nên kết quả ra 1,0 cần nhân 255

def reconstruct_by_dilation(marker_img, kernel, mask_img):
    D_pre = marker_img
    while True:
        dilated_img = dilate(D_pre,kernel)
        D_after = bitwise_and(dilated_img,mask_img)
        if(np.all(D_pre==D_after)):
            return D_pre
        D_pre = D_after
    return D_pre

def reconstruct_by_erosion(marker_img, kernel, mask_img):
    E_pre = marker_img
    while True:
        E_after = erode(E_pre,kernel) + mask_img
        if(np.all(E_after==E_pre)):
            return E_pre
        E_pre = E_after
    return E_pre

def open_by_reconstruction(marker_img, kernel, mask_img, iteration):
    tmp_before = marker_img
    for i in range(iteration):
        tmp_after = erode(tmp_before,kernel)
        tmp_before = tmp_after
    result = reconstruct_by_dilation(tmp_before,kernel,mask_img)
    return result

def close_by_reconstruction(marker_img, kernel, mask_img, iteration):
    tmp_before = marker_img
    for i in range(iteration):
        tmp_after = dilate(tmp_before, kernel)
        tmp_before = tmp_after
    result = reconstruct_by_erosion(tmp_before, kernel,mask_img)
    return result

def fill_hole(img,kernel):
    img_shape = img.shape
    invert_img = np.bitwise_not(img)
    marker_img = np.zeros((img_shape[0], img_shape[1]), np.uint8)

    marker_img[:,0] = 255 - img[:,0]
    marker_img[:,img_shape[1]-1] = 255-img[:,img_shape[1]-1]

    marker_img[0, :]=255-img[0,:]
    marker_img[img_shape[0]-1,:] = 255- img[img_shape[0]-1,:]

    H = reconstruct_by_dilation(marker_img,kernel,invert_img)

    return np.bitwise_not(H)

def extract_connected_components(img, kernel):
    img_shape = img.shape
    labels = np.zeros((img_shape[0],img_shape[1]), np.int)
    tmp=img
    number_cc = []
    number_pixel = []
    k=0
    while(np.any(tmp!=0)):
        current_label=labels.max()
        k+=1
        flag=0
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if(tmp[i,j]==255):
                    x,y = i,j
                    flag=1
                    break
            if(flag==1):
                break
        A = np.zeros((img_shape[0],img_shape[1]))
        A[x,y]=255
        while True:
            B = bitwise_and(dilate(A,kernel),img)
            if(np.all(A==B)):
                break
            A = B
        number_cc.append(k)
        number_pixel.append(int(A.sum()/255))
        tmp = tmp-A
        labels[A!=0]=current_label+1
    info = dict(zip(number_cc, number_pixel))
    return info, labels

def convex_hull(img):
    set_4_kernel = []
    set_4_kernel.append(np.array([[1,0,0],[1,-1,0],[1,0,0]]))
    set_4_kernel.append(np.array([[1,1,1],[0,-1,0],[0,0,0]]))
    set_4_kernel.append(np.array([[0,0,1],[0,-1,1],[0,0,1]]))
    set_4_kernel.append(np.array([[0,0,0],[0,-1,0],[1,1,1]]))
    result = []
    for kernel in set_4_kernel:
        tmp_before = img
        while True:
            tmp_after = hitmiss(tmp_before,kernel) + tmp_before
            if(np.all(tmp_before==tmp_after)):
                break
            tmp_before = tmp_after
        result.append(tmp_before)

    return result[0] + result[1] +result[2] +result[3]

def thin(img):
    set_8_kernel = []
    set_8_kernel.append(np.array([[-1,-1,-1],[0,1,0],[1,1,1]]))
    set_8_kernel.append(np.array([[0,-1,-1],[1,1,-1],[1,1,0]]))
    set_8_kernel.append(np.array([[1,0,-1],[1,1,-1],[1,0,-1]]))
    set_8_kernel.append(np.array([[1,1,0],[1,1,-1],[0,-1,-1]]))
    set_8_kernel.append(np.array([[1,1,1],[0,1,0],[-1,-1,-1]]))
    set_8_kernel.append(np.array([[0,1,1],[-1,1,1],[-1,-1,0]]))
    set_8_kernel.append(np.array([[-1,0,1],[-1,1,1],[-1,0,1]]))
    set_8_kernel.append(np.array([[-1,-1,0],[-1,1,1],[0,1,1]]))

    tmp1 = img-hitmiss(img, set_8_kernel[0])
    tmp2 = tmp1-hitmiss(tmp1, set_8_kernel[1])
    tmp3 = tmp2-hitmiss(tmp2, set_8_kernel[2])
    tmp4 = tmp3-hitmiss(tmp3, set_8_kernel[3])
    tmp5 = tmp4-hitmiss(tmp4, set_8_kernel[4])
    tmp6 = tmp5-hitmiss(tmp5, set_8_kernel[5])
    tmp7 = tmp6-hitmiss(tmp6, set_8_kernel[6])
    tmp8 = tmp7-hitmiss(tmp7, set_8_kernel[7])
    tmp_before = tmp8
    while True:
        for kernel in set_8_kernel:
            tmp_after = tmp_before - hitmiss(tmp_before, kernel)
            if(np.all(tmp_before==tmp_after)):
                return tmp_before
            tmp_before = tmp_after
    return tmp_before

def thicken(img):
    # Cách này cho ra kết quả mở rộng gần như toàn bức ảnh
    # Do thinning lặp đến mức foreground còn bé nhất
    '''tmp = 255-img
    result = 255 - thin(tmp)
    return result'''
    # Chuyển sang chỉ dùng 8 kernel 1 lần duy nhất chứ không lặp
    set_8_kernel = []
    set_8_kernel.append(np.array([[1,1,1],[0,-1,0],[-1,-1,-1]]))
    set_8_kernel.append(np.array([[0,1,1],[-1,-1,1],[-1,-1,0]]))
    set_8_kernel.append(np.array([[-1,0,1],[-1,-1,1],[-1,0,1]]))
    set_8_kernel.append(np.array([[-1,-1,0],[-1,-1,1],[0,1,1]]))
    set_8_kernel.append(np.array([[-1,-1,-1],[0,-1,0],[1,1,1]]))
    set_8_kernel.append(np.array([[0,-1,-1],[1,-1,-1],[1,1,0]]))
    set_8_kernel.append(np.array([[1,0,-1],[1,-1,-1],[1,0,-1]]))
    set_8_kernel.append(np.array([[1,1,0],[1,-1,-1],[0,-1,-1]]))
    tmp1 = img + hitmiss(img, set_8_kernel[0])
    tmp2 = tmp1 + hitmiss(tmp1, set_8_kernel[1])
    tmp3 = tmp2 + hitmiss(tmp2, set_8_kernel[2])
    tmp4 = tmp3 + hitmiss(tmp3, set_8_kernel[3])
    tmp5 = tmp4 + hitmiss(tmp4, set_8_kernel[4])
    tmp6 = tmp5 + hitmiss(tmp5, set_8_kernel[5])
    tmp7 = tmp6 + hitmiss(tmp6, set_8_kernel[6])
    tmp8 = tmp7 + hitmiss(tmp7, set_8_kernel[7])
    
    return tmp8
def skeleton(img, kernel):
    A_erode_kB_before = img
    A_erode_kB_open_B_before = open(A_erode_kB_before,kernel)
    Sk_A_before = A_erode_kB_before - A_erode_kB_open_B_before
    sum_Sk_A = Sk_A_before

    while True:
        A_erode_kB_after = erode(A_erode_kB_before,kernel)
        if(np.all(A_erode_kB_after)==0):
            break
        A_erode_kB_open_B_after = open(A_erode_kB_after, kernel)
        Sk_A_after = A_erode_kB_after - A_erode_kB_open_B_after
        sum_Sk_A += Sk_A_after
        A_erode_kB_before = A_erode_kB_after
    return sum_Sk_A

def prun(img):
    set_8_kernel = []
    set_8_kernel.append(np.array([[0,-1,-1],[1,1,-1],[0,-1,-1]]))
    set_8_kernel.append(np.array([[0,1,0],[-1,1,-1],[-1,-1,-1]]))
    set_8_kernel.append(np.array([[-1,-1,0],[-1,1,1],[-1,-1,0]]))
    set_8_kernel.append(np.array([[-1,-1,-1],[-1,1,-1],[0,1,0]]))
    set_8_kernel.append(np.array([[1,-1,-1],[-1,1,-1],[-1,-1,-1]]))
    set_8_kernel.append(np.array([[-1,-1,1],[-1,1,-1],[-1,-1,-1]]))
    set_8_kernel.append(np.array([[-1,-1,-1],[-1,1,-1],[-1,-1,1]]))
    set_8_kernel.append(np.array([[-1,-1,-1],[-1,1,-1],[1,-1,-1]]))
    tmp1 = img-hitmiss(img, set_8_kernel[0])
    tmp2 = tmp1-hitmiss(tmp1, set_8_kernel[1])
    tmp3 = tmp2-hitmiss(tmp2, set_8_kernel[2])
    tmp4 = tmp3-hitmiss(tmp3, set_8_kernel[3])
    tmp5 = tmp4-hitmiss(tmp4, set_8_kernel[4])
    tmp6 = tmp5-hitmiss(tmp5, set_8_kernel[5])
    tmp7 = tmp6-hitmiss(tmp6, set_8_kernel[6])
    tmp8 = tmp7-hitmiss(tmp7, set_8_kernel[7])
    
    X1 = tmp8
    X1_before = X1
    X2 = np.zeros_like(img)
    for kernel in set_8_kernel:
        X1_after = hitmiss(X1_before,kernel)
        X2+=X1_after
        X1_before = X1_after

    H = np.ones((3,3))
    X3 = bitwise_and(dilate(X2,H),img)
    return X1 + X3


