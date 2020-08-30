import sys
import getopt
import cv2
import numpy as np
from morphological_operator import binary
from morphological_operator import grayscale

def operator(in_file,out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow('binary image', img)
    cv2.waitKey(wait_key_time)
  
    kernel = np.ones((5, 5), np.uint8)

    img_out = None

    '''
    TODO: implement morphological operators
    '''
    if mor_op == 'dilate':
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation

    elif mor_op == 'erode':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual
    
    elif mor_op == 'open':
        img_opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = binary.open(img, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual

    elif mor_op == 'close':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV closing image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = binary.close(img, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual

    elif mor_op == 'hitmiss':
        kernel_hitmiss=np.array([[0,1,0],[1,-1,1],[0,1,0]])

        img_hitmiss = cv2.morphologyEx(img,cv2.MORPH_HITMISS,kernel_hitmiss)
        cv2.imshow('OpenCV Hit-or-Miss image', img_hitmiss)
        cv2.waitKey(wait_key_time)

        img_hitmiss_manual = binary.hitmiss(img,kernel_hitmiss)
        cv2.imshow('manual Hit-or-Miss image', img_hitmiss_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hitmiss_manual


    elif mor_op == 'boundary':
        img_boundary = img-cv2.erode(img,kernel)
        cv2.imshow('OpenCV boundary extraction image', img_boundary)
        cv2.waitKey(wait_key_time)

        img_boundary_manual = binary.extract_boundary(img,kernel)
        cv2.imshow('manual boundary extraction image', img_boundary_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_boundary_manual
    
    elif mor_op == 'fillhole':
        #OpenCV
        img_floodfill = img.copy()
        cv2.floodFill(img_floodfill, None, (0,0), 255)
        img_floodfill_inv = cv2.bitwise_not(img_floodfill)
        img_hole_filling = img+img_floodfill_inv
        cv2.imshow('OpenCV hole filling image', img_hole_filling)
        cv2.waitKey(wait_key_time)

        #manual
        img_hole_filling_manual = binary.fill_hole(img,kernel)
        cv2.imshow('manual hole filling image',img_hole_filling_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hole_filling_manual

    elif mor_op == 'connectedcomponents':
        num, labels = cv2.connectedComponents(img, connectivity=8)
        img_labels_show = imshow_components(labels)
        cv2.imshow('OpenCV connected components image', img_labels_show)
        cv2.waitKey(wait_key_time)

        info, labels_manual = binary.extract_connected_components(img, kernel)
        img_labels_show_manual = imshow_components(labels_manual)

        for i in info:
            print('Number pixels of connected component {} : {}'.format(i, info[i]))
        cv2.imshow('manual connected components image', img_labels_show_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_labels_show_manual

    elif mor_op=='convexhull':
        img_convex_hull_manual = binary.convex_hull(img)
        cv2.imshow('manual convex hull image', img_convex_hull_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_convex_hull_manual

    elif mor_op=='thin':
        img_thinning = cv2.ximgproc.thinning(img)
        cv2.imshow('OpenCV thinning image', img_thinning)
        cv2.waitKey(wait_key_time)
        
        img_thinning_manual = binary.thin(img)
        cv2.imshow('manual thinning image', img_thinning_manual)
        cv2.waitKey(wait_key_time)
        
        img_out = img_thinning_manual
    
    elif mor_op=='thicken':
        img_thickening_manual = binary.thicken(img)
        cv2.imshow('manual thickening image', img_thickening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thickening_manual
    
    elif mor_op == 'skeleton':
        img_skeleton_manual = binary.skeleton(img,kernel)
        cv2.imshow('manual skeleton image', img_skeleton_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_skeleton_manual

    elif mor_op == 'prun':
        img_pruning_manual = binary.prun(img)
        cv2.imshow('manual pruning image', img_pruning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_pruning_manual

    # Grayscale image

    elif mor_op == 'erode_gray':
        img_erosion = cv2.morphologyEx(img_gray,cv2.MORPH_ERODE,kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = grayscale.erode(img_gray, kernel)
        cv2.imshow('manual ersion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual

    elif mor_op=='dilate_gray':
        img_dilation = cv2.morphologyEx(img_gray,cv2.MORPH_DILATE,kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = grayscale.dilate(img_gray, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual
    
    elif mor_op=='open_gray':
        img_opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = grayscale.open(img_gray, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual

    elif mor_op=='close_gray':
        
        img_closing = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV closeing image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = grayscale.close(img_gray, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual
    
    elif mor_op=='smooth_gray':
        img_opening =cv2.morphologyEx(img_gray,cv2.MORPH_OPEN,kernel)
        img_smoothing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV smoothing image', img_smoothing)
        cv2.waitKey(wait_key_time)

        img_smoothing_manual = grayscale.smooth(img_gray,kernel)
        cv2.imshow('manual smoothing image', img_smoothing_manual)
        cv2.waitKey(wait_key_time)

        img_out=img_smoothing_manual
    
    elif mor_op=='gradient_gray':
        img_gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow('OpenCV gradient image', img_gradient)
        cv2.waitKey(wait_key_time)

        img_gradient_manual = grayscale.gradient(img_gray,kernel)
        cv2.imshow('manual gradient image', img_gradient_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gradient_manual

    elif mor_op=='tophat_gray':
        img_top_hat= cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow('OpenCV top-hat image', img_top_hat)
        cv2.waitKey(wait_key_time)

        img_top_hat_manual = grayscale.top_hat(img_gray,kernel)
        cv2.imshow('manual top-hat image',img_top_hat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_top_hat_manual

    elif mor_op=='bottomhat_gray':
        img_bottom_hat_manual = grayscale.bottom_hat(img_gray,kernel)
        cv2.imshow('manual bottom-hat image',img_bottom_hat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_bottom_hat_manual
    
    elif mor_op=='granulometry_gray':
        img_granulometry_manual = grayscale.granulometry(img_gray,kernel)
        cv2.imshow('manual granulometry image', img_granulometry_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_granulometry_manual

    if img_out is not None:
        cv2.imwrite(out_file, img_out)

def operator_reconstruction(in_file,marker_file,iteration, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow('binary image', img)
    cv2.waitKey(wait_key_time)
  
    kernel = np.ones((5, 5), np.uint8)

    img_out = None

    # Marker
    marker_origin = cv2.imread(marker_file)


    marker_gray  = cv2.imread(marker_file,0)  
    cv2.imshow('gray marker image', marker_gray)
    cv2.waitKey(wait_key_time)

    (thresh_re, marker) = cv2.threshold(marker_gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary marker image', marker)
    cv2.waitKey(wait_key_time)

    if mor_op == 'reconstruct_by_dilation':
        img_reconstruct_by_dialtion = binary.reconstruct_by_dilation(marker, kernel,img)
        cv2.imshow('manual reconstruction by dilation', img_reconstruct_by_dialtion)
        cv2.waitKey(wait_key_time)

        img_out = img_reconstruct_by_dialtion

    elif mor_op == 'reconstruct_by_erosion':
        img_reconstruct_by_erosion = binary.reconstruct_by_erosion(marker,kernel,img)
        cv2.imshow('manual reconstruction by erosion', img_reconstruct_by_erosion)
        cv2.waitKey(wait_key_time)
        
        img_out = img_reconstruct_by_erosion
    
    elif mor_op=='open_by_reconstruction':
        img_open_by_reconstruction = binary.open_by_reconstruction(marker,kernel,img, iteration)
        cv2.imshow('manual opening by reconstruction image',img_open_by_reconstruction)
        cv2.waitKey(wait_key_time)

        img_out = img_open_by_reconstruction
    
    elif mor_op=='close_by_reconstruction':
        img_close_by_reconstruction = binary.close_by_reconstruction(marker, kernel, img, iteration)
        cv2.imshow('manual closing by reconstruction',img_close_by_reconstruction)
        cv2.waitKey(wait_key_time)

        img_out = img_close_by_reconstruction

    elif mor_op =='reconstruct_by_dilation_gray':
        img_reconstruct_by_dialtion_gray = grayscale.reconstruct_by_dilation(marker_gray, kernel, img_gray)
        cv2.imshow('manual reconstruction by dilation',img_reconstruct_by_dialtion_gray)
        cv2.waitKey(wait_key_time)

        img_out = img_reconstruct_by_dialtion_gray
    
    elif mor_op=='reconstruct_by_erosion_gray':
        img_reconstruct_by_erosion_gray = grayscale.reconstruct_by_erosion(marker_gray, kernel, img_gray)
        cv2.imshow('manual reconstruction by dilation',img_reconstruct_by_erosion_gray)
        cv2.waitKey(wait_key_time)

        img_out = img_reconstruct_by_erosion_gray

    elif mor_op=='open_by_reconstruction_gray':
        img_open_by_reconstruction_gray = grayscale.open_by_reconstruction(marker_gray,kernel,img_gray,iteration)
        cv2.imshow('manual opening by reconstruction',img_open_by_reconstruction_gray)
        
        cv2.waitKey(wait_key_time)

        img_out = img_open_by_reconstruction_gray

    elif mor_op=='close_by_reconstruction_gray':
        img_open_by_reconstruction_gray = grayscale.close_by_reconstruction(marker_gray,kernel,img_gray,iteration)
        cv2.imshow('manual opening by reconstruction',img_open_by_reconstruction_gray)
        cv2.waitKey(wait_key_time)

        img_out = img_open_by_reconstruction_gray


    if img_out is not None:
        cv2.imwrite(out_file, img_out)


# Hien thi anh cac thanh phan lien thong opencv
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img

def main(argv):
    input_file = ''
    marker_file = ''
    iteration=0
    output_file = ''
    mor_op = ''
    wait_key_time = 0
    argc = len(sys.argv)

    description = 'main.py -i <input_file> -m <marker>(for reconstruct) -k <iteration>(for open close reconstruct) -o <output_file> -p <mor_operator> -t <wait_key_time>'
    try:
        opts = ''
        if(argc==9):
            opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
        elif(argc==11):
            opts, args = getopt.getopt(argv, "hi:m:o:p:t:", ["in_file=", "marker_file=","out_file=", "mor_operator=", "wait_key_time="])
        elif(argc==13):
            opts, args = getopt.getopt(argv, "hi:m:k:o:p:t:", ["in_file=", "marker_file=","iteration=","out_file=", "mor_operator=", "wait_key_time="])
        else:
            print(description)
    except getopt.GetoptError:
        print(description)
        sys.exit(2)
    if(argc==9):
        for opt, arg in opts:
            if opt == '-h':
                print(description)
                sys.exit()
            elif opt in ("-i", "--in_file"):
                input_file = arg
            elif opt in ("-o", "--out_file"):
                output_file = arg
            elif opt in ("-p", "--mor_operator"):
                mor_op = arg
            elif opt in ("-t", "--wait_key_time"):
                wait_key_time = int(arg)

        print('Input file is ', input_file)
        print('Output file is ', output_file)
        print('Morphological operator is ', mor_op)
        print('Wait key time is ', wait_key_time)

        operator(input_file, output_file, mor_op, wait_key_time)
        cv2.waitKey(wait_key_time)

    elif(argc==11):
        for opt, arg in opts:
            if opt == '-h':
                print(description)
                sys.exit()
            elif opt in ("-i", "--in_file"):
                input_file = arg
            elif opt in ("-m", "--marker_file"):
                marker_file = arg
            elif opt in ("-o", "--out_file"):
                output_file = arg
            elif opt in ("-p", "--mor_operator"):
                mor_op = arg
            elif opt in ("-t", "--wait_key_time"):
                wait_key_time = int(arg)

        print('Input file is ', input_file)
        print('Marker file is ',marker_file)
        print('Output file is ', output_file)
        print('Morphological operator is ', mor_op)
        print('Wait key time is ', wait_key_time)

        operator_reconstruction(input_file, marker_file,0, output_file, mor_op, wait_key_time)
        cv2.waitKey(wait_key_time)
    elif(argc==13):
        iteration = 0
        for opt, arg in opts:
            if opt == '-h':
                print(description)
                sys.exit()
            elif opt in ("-i", "--in_file"):
                input_file = arg
            elif opt in ("-m", "--marker_file"):
                marker_file = arg
            elif opt in("-k", "--iteration"):
                iteration = int(arg)
            elif opt in ("-o", "--out_file"):
                output_file = arg
            elif opt in ("-p", "--mor_operator"):
                mor_op = arg
            elif opt in ("-t", "--wait_key_time"):
                wait_key_time = int(arg)

        print('Input file is ', input_file)
        print('Marker file is ',marker_file)
        print('Iteration is ',iteration)
        print('Output file is ', output_file)
        print('Morphological operator is ', mor_op)
        print('Wait key time is ', wait_key_time)

        operator_reconstruction(input_file, marker_file,iteration, output_file, mor_op, wait_key_time)
        cv2.waitKey(wait_key_time)



if __name__ == "__main__":
    main(sys.argv[1:])
