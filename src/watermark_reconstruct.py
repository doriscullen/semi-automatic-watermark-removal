import numpy as np
import cv2
import os
import scipy
from scipy.sparse import *
from scipy.sparse import linalg
from src.estimate_watermark import *
from src.closed_form_matting import *
from numpy import nan, isnan

def get_cropped_images(foldername, num_images, num_w, start, length, shape):
    '''
    This is the part where we get all the images, extract their parts, and then add it to our matrix
    '''
    Num = num_images*num_w
    images_cropped = np.zeros((Num,) + shape)
    # get images
    # Store all the watermarked images
    # start, and end are already stored
    # just crop and store images
    file_names = []
    _s, _l = start, length
    index = 0

    # Iterate over all images
    for r, dirs, files in os.walk(foldername):
        im = np.zeros([600,600,3]).astype( np.uint8 )
        
        for file in files:
            _img = cv2.imread(os.sep.join([r, file]))
            im[150:450,150:450] = _img[0:300,0:300]
            if _img is not None:
                # estimate the watermark part
                file_names.append(file)
                for i in range(num_w):
                    images_cropped[index, :, :, :] = im[_s[i][0]:(_s[i][0]+_l[i][0]), _s[i][1]:(_s[i][1]+_l[i][1])]
                    index+=1
                    # print(tmp.shape())
                    
                if index>= Num:
                    return (images_cropped, file_names)
            else:
                print("%s not found."%(file))

    return (images_cropped, file_names)

def get_cropped_images2(foldername, num_images, num_w, start, length, shape):
    '''
    This is the part where we get all the images, extract their parts, and then add it to our matrix
    '''
    Num = num_images*num_w
    images_cropped = np.zeros((Num,) + shape)
    # get images
    # Store all the watermarked images
    # start, and end are already stored
    # just crop and store images
    file_names = []
    _s, _l = start, length
    index = 0

    # Iterate over all images
    for i in range(num_images):
        _img = cv2.imread(foldername+str(i+1)+"_random.png")
        if _img is not None:
            # estimate the watermark part
            if _s[i][0]>0 and _s[i][1]>0 and (_s[i][0]+_l[i][0])<300 and (_s[i][1]+_l[i][1])<300:
                file_names.append(str(i+1)+"_random.png")
                images_cropped[index, :, :, :] = _img[_s[i][0]:(_s[i][0]+_l[i][0]), _s[i][1]:(_s[i][1]+_l[i][1])]
                index+=1
#             print(i)
#             print(_s[i])
#             plt.imshow(PlotImage(images_cropped[i]))
#             plt.draw()
        else:
            print("%s not found."%(file))

    return (images_cropped, file_names, index)

# get sobel coordinates for y
def _get_ysobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
        (i-1, j, k, -2), (i-1, j-1, k, -1), (i-1, j+1, k, -1),
        (i+1, j, k,  2), (i+1, j-1, k,  1), (i+1, j+1, k,  1)
    ]

# get sobel coordinates for x
def _get_xsobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
        (i, j-1, k, -2), (i-1, j-1, k, -1), (i-1, j+1, k, -1),
        (i, j+1, k,  2), (i+1, j-1, k,  1), (i+1, j+1, k,  1)
    ]

# filter
def _filter_list_item(coord, shape):
    i, j, k, v = coord
    m, n, p = shape
    if i>=0 and i<m and j>=0 and j<n:
        return True

# Change to ravel index
# also filters the wrong guys
def _change_to_ravel_index(li, shape):
    li = filter(lambda x: _filter_list_item(x, shape), li)
    i, j, k, v = zip(*li) # i=[i1,i2...]
    return zip(np.ravel_multi_index((i, j, k), shape), v) #[(index,v),()...]

# TODO: Consider wrap around of indices to remove the edge at the end of sobel
# get Sobel sparse matrix for Y
def get_ySobel_matrix(m, n, p):
    size = m*n*p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))   #[(i,j,k),()...]
    ijk_nbrs = map(lambda x: _get_ysobel_coord(x, shape), ijk)   #[(i,j,k,v),()...]
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs) # [(index,v),()...]
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)  # i:像素序号 j:扁平矩阵中的index vals:值
    return coo_matrix((vals, (i, j)), shape=(size, size))


# get Sobel sparse matrix for X
def get_xSobel_matrix(m, n, p):
    size = m*n*p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))                      #把i,j,k打包成元组(i,j,k)
    ijk_nbrs = map(lambda x: _get_xsobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))

# get estimated normalized alpha matte
def estimate_normalized_alpha(J, W_m, num_images=1, threshold=170, invert=False, adaptive=False, adaptive_threshold=21, c2=10):
    _Wm = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8) #unit8: 无符号整数（0 to 255）
    
    if adaptive:    # 自适应阈值二值化
        thr = cv2.adaptiveThreshold(_Wm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_threshold, c2)
    else:   # 固定阈值二值化
        ret, thr = cv2.threshold(_Wm, threshold, 255, cv2.THRESH_BINARY)

    #-----------
    #print("thr in estimate_normalized_alpha")
    #print(thr)
    
    if invert:
        thr = 255-thr
    thr = np.stack([thr, thr, thr], axis=2)
    
    #-----------
    #print("thr in estimate_normalized_alpha, after np.stack")
    #print(thr)

    num, m, n, p = J.shape
    alpha = np.zeros((num_images, m, n))
    # iterpatch = 900  # 这是个啥？

    print("Estimating normalized alpha using %d images."%(num_images))
    # for all images, calculate alpha
    for idx in range(num_images):
        print("estimate_normalized_alpha"+str(idx))
        imgcopy = thr
        alph = closed_form_matte(J[idx], imgcopy)   # closed_form_matting.py
        alpha[idx] = alph

    alpha = np.median(alpha, axis=0)
    return alpha

def estimate_blend_factor(J, W_m, alph, threshold=0.01*255):   # 论文中的依据：Jk = Est(Wm) + c · αn · E[Ik]
    K, m, n, p = J.shape
    Jm = (J - W_m)
    gx_jm = np.zeros(J.shape)
    gy_jm = np.zeros(J.shape)

    for i in range(K):
        gx_jm[i] = cv2.Sobel(Jm[i], cv2.CV_64F, 1, 0, 3)
        gy_jm[i] = cv2.Sobel(Jm[i], cv2.CV_64F, 0, 1, 3)

    Jm_grad = np.sqrt(gx_jm**2 + gy_jm**2)

    est_Ik = alph*np.median(J, axis=0)                   # 论文3.2节高亮部分
    gx_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 1, 0, 3)
    gy_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 0, 1, 3)
    estIk_grad = np.sqrt(gx_estIk**2 + gy_estIk**2)

    C = []   # 论文中好像说用least squares（最小二乘法）
    for i in range(3):
        c_i = np.sum(Jm_grad[:,:,:,i]*estIk_grad[:,:,i])/np.sum(np.square(estIk_grad[:,:,i]))/K
        print(c_i)
        C.append(c_i)

    return C, est_Ik


def Func_Phi(X, epsilon=1e-3):            #phi(S^2) = sqrt( s^2 + epsilon^2)
    return np.sqrt(X + epsilon**2)

def Func_Phi_deriv(X, epsilon=1e-3):
    return 0.5/Func_Phi(X, epsilon)

def solve_images(J, W_m, alpha, W_init, gamma=1, beta=1, lambda_w=0.005, lambda_i=1, lambda_a=0.01, iters=2):
    '''
    Master solver, follows the algorithm given in the supplementary.
    W_init: Initial value of W
    Step 1: Image Watermark decomposition
    '''
    # prepare variables
    K, m, n, p = J.shape
    size = m*n*p

    sobelx = get_xSobel_matrix(m, n, p)   # shape=(size, size)
    sobely = get_ySobel_matrix(m, n, p)   # shape=(size, size)
    Ik = np.zeros(J.shape)
    Wk = np.zeros(J.shape)
    for i in range(K):
        Ik[i] = J[i] - W_m
        Wk[i] = W_init.copy()

    # This is for median images
    W = W_init.copy()

    # Iterations
    for _ in range(iters):

        print("------------------------------------")
        print("Iteration: %d"%(_))

        # Step 1
        print("Step 1")
        alpha_gx = cv2.Sobel(alpha, cv2.CV_64F, 1, 0, 3) #CV_64FC1: 64F代表每一个像素点元素占64位浮点数，通道数为1
        alpha_gy = cv2.Sobel(alpha, cv2.CV_64F, 0, 1, 3)

        Wm_gx = cv2.Sobel(W_m, cv2.CV_64F, 1, 0, 3)  #CV_64FC3? 应该还是三通道的
        Wm_gy = cv2.Sobel(W_m, cv2.CV_64F, 0, 1, 3)

        cx = diags(np.abs(alpha_gx).reshape(-1))  # .reshape(-1)变成一维数组
        cy = diags(np.abs(alpha_gy).reshape(-1))

        alpha_diag = diags(alpha.reshape(-1))
        alpha_bar_diag = diags((1-alpha).reshape(-1))

        for i in range(K):
            # prep vars
            Wkx = cv2.Sobel(Wk[i], cv2.CV_64F, 1, 0, 3)
            Wky = cv2.Sobel(Wk[i], cv2.CV_64F, 0, 1, 3)

            Ikx = cv2.Sobel(Ik[i], cv2.CV_64F, 1, 0, 3)
            Iky = cv2.Sobel(Ik[i], cv2.CV_64F, 0, 1, 3)

            alphaWk = alpha*Wk[i]
            alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)
            alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)        

            phi_data = diags( Func_Phi_deriv(np.square(alpha*Wk[i] + (1-alpha)*Ik[i] - J[i]).reshape(-1)) )
            phi_W = diags( Func_Phi_deriv(np.square( np.abs(alpha_gx)*Wkx + np.abs(alpha_gy)*Wky  ).reshape(-1)) )
            phi_I = diags( Func_Phi_deriv(np.square( np.abs(alpha_gx)*Ikx + np.abs(alpha_gy)*Iky  ).reshape(-1)) )
            phi_f = diags( Func_Phi_deriv( ((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2 ).reshape(-1)) )
            phi_aux = diags( Func_Phi_deriv(np.square(Wk[i] - W).reshape(-1)) )
            phi_rI = diags( Func_Phi_deriv( np.abs(alpha_gx)*(Ikx**2) + np.abs(alpha_gy)*(Iky**2) ).reshape(-1) )
            phi_rW = diags( Func_Phi_deriv( np.abs(alpha_gx)*(Wkx**2) + np.abs(alpha_gy)*(Wky**2) ).reshape(-1) )

            L_i = sobelx.T.dot(cx*phi_rI).dot(sobelx) + sobely.T.dot(cy*phi_rI).dot(sobely)
            L_w = sobelx.T.dot(cx*phi_rW).dot(sobelx) + sobely.T.dot(cy*phi_rW).dot(sobely)
            L_f = sobelx.T.dot(phi_f).dot(sobelx) + sobely.T.dot(phi_f).dot(sobely)
            A_f = alpha_diag.T.dot(L_f).dot(alpha_diag) + gamma*phi_aux

            bW = alpha_diag.dot(phi_data).dot(J[i].reshape(-1)) + beta*L_f.dot(W_m.reshape(-1)) + gamma*phi_aux.dot(W.reshape(-1))
            bI = alpha_bar_diag.dot(phi_data).dot(J[i].reshape(-1))

            A = vstack([hstack([(alpha_diag**2)*phi_data + lambda_w*L_w + beta*A_f, alpha_diag*alpha_bar_diag*phi_data]), \
                         hstack([alpha_diag*alpha_bar_diag*phi_data, (alpha_bar_diag**2)*phi_data + lambda_i*L_i])]).tocsr()

            b = np.hstack([bW, bI])
            x = linalg.spsolve(A, b)
            
            Wk[i] = x[:size].reshape(m, n, p)
            Ik[i] = x[size:].reshape(m, n, p)
            # plt.subplot(3,1,1); plt.imshow(PlotImage(J[i]))
            # plt.subplot(3,1,2); plt.imshow(PlotImage(Wk[i]))
            # plt.subplot(3,1,3); plt.imshow(PlotImage(Ik[i]))
            # plt.draw()
            # plt.pause(0.001)
            print(i)

        # Step 2
        print("Step 2")
        W = np.median(Wk, axis=0)

        # plt.imshow(PlotImage(W))
        # plt.draw()
        # plt.pause(0.001)
        
        # Step 3
        print("Step 3")
        W_diag = diags(W.reshape(-1))
        
        for i in range(K):
            alphaWk = alpha*Wk[i]
            alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)  # Dx
            alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)  # Dy   
            phi_f = diags( Func_Phi_deriv( ((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2 ).reshape(-1)) )
            
            phi_kA = diags(( (Func_Phi_deriv((((alpha*Wk[i] + (1-alpha)*Ik[i] - J[i])**2)))) * ((W-Ik[i])**2)  ).reshape(-1))  # phi'k
            phi_kB = (( (Func_Phi_deriv((((alpha*Wk[i] + (1-alpha)*Ik[i] - J[i])**2))))*(W-Ik[i])*(J[i]-Ik[i])  ).reshape(-1))  # Ak(J-Ik)

            phi_alpha = diags(Func_Phi_deriv(alpha_gx**2 + alpha_gy**2).reshape(-1))
            L_alpha = sobelx.T.dot(phi_alpha.dot(sobelx)) + sobely.T.dot(phi_alpha.dot(sobely))

            L_f = sobelx.T.dot(phi_f).dot(sobelx) + sobely.T.dot(phi_f).dot(sobely)
            A_tilde_f = W_diag.T.dot(L_f).dot(W_diag)
            # Ax = b, setting up A
            if i==0:
                A1 = phi_kA + lambda_a*L_alpha + beta*A_tilde_f
                b1 = phi_kB + beta*W_diag.dot(L_f).dot(W_m.reshape(-1))
            else:
                A1 += (phi_kA + lambda_a*L_alpha + beta*A_tilde_f)
                b1 += (phi_kB + beta*W_diag.T.dot(L_f).dot(W_m.reshape(-1)))

        alpha = linalg.spsolve(A1, b1).reshape(m,n,p)

        # plt.imshow(PlotImage(alpha))
        # plt.draw()
        # plt.pause(0.001)
    
    return (Wk, Ik, W, alpha)


def changeContrastImage(J, I):
    cJ1 = J[0, 0, :]
    cJ2 = J[-1, -1, :]

    cI1 = I[0, 0, :]
    cI2 = I[-1,-1, :]

    I_m = cJ1 + (I-cI1)/(cI2-cI1)*(cJ2-cJ1)
    return I_m



def get_images(J, W, alpha, gamma=1, beta=1, lambda_w=0.005, lambda_i=1, lambda_a=0.01, iters=2):
    # prepare variables
    K, m, n, p = J.shape
    size = m*n*p

    sobelx = get_xSobel_matrix(m, n, p)   # shape=(size, size)
    sobely = get_ySobel_matrix(m, n, p)   # shape=(size, size)
    Ik = np.zeros(J.shape)
    Wk = np.zeros(J.shape)
    for i in range(K):
        Ik[i] = J[i] - W
        Wk[i] = W.copy()
       
    print("get the original images")
    alpha_gx = cv2.Sobel(alpha, cv2.CV_64F, 1, 0, 3) #CV_64FC1: 64F代表每一个像素点元素占64位浮点数，通道数为1
    alpha_gy = cv2.Sobel(alpha, cv2.CV_64F, 0, 1, 3)

    Wm_gx = cv2.Sobel(W, cv2.CV_64F, 1, 0, 3)  #CV_64FC3? 应该还是三通道的
    Wm_gy = cv2.Sobel(W, cv2.CV_64F, 0, 1, 3)

    cx = diags(np.abs(alpha_gx).reshape(-1))  # .reshape(-1)变成一维数组
    cy = diags(np.abs(alpha_gy).reshape(-1))

    alpha_diag = diags(alpha.reshape(-1))
    alpha_bar_diag = diags((1-alpha).reshape(-1))

    for i in range(K):
        # prep vars
        Wkx = cv2.Sobel(Wk[i], cv2.CV_64F, 1, 0, 3)
        Wky = cv2.Sobel(Wk[i], cv2.CV_64F, 0, 1, 3)

        Ikx = cv2.Sobel(Ik[i], cv2.CV_64F, 1, 0, 3)
        Iky = cv2.Sobel(Ik[i], cv2.CV_64F, 0, 1, 3)

        alphaWk = alpha*Wk[i]
        alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)
        alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)        

        phi_data = diags( Func_Phi_deriv(np.square(alpha*Wk[i] + (1-alpha)*Ik[i] - J[i]).reshape(-1)) )
        phi_W = diags( Func_Phi_deriv(np.square( np.abs(alpha_gx)*Wkx + np.abs(alpha_gy)*Wky  ).reshape(-1)) )
        phi_I = diags( Func_Phi_deriv(np.square( np.abs(alpha_gx)*Ikx + np.abs(alpha_gy)*Iky  ).reshape(-1)) )
        phi_f = diags( Func_Phi_deriv( ((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2 ).reshape(-1)) )
        phi_aux = diags( Func_Phi_deriv(np.square(Wk[i] - W).reshape(-1)) )
        phi_rI = diags( Func_Phi_deriv( np.abs(alpha_gx)*(Ikx**2) + np.abs(alpha_gy)*(Iky**2) ).reshape(-1) )
        phi_rW = diags( Func_Phi_deriv( np.abs(alpha_gx)*(Wkx**2) + np.abs(alpha_gy)*(Wky**2) ).reshape(-1) )

        L_i = sobelx.T.dot(cx*phi_rI).dot(sobelx) + sobely.T.dot(cy*phi_rI).dot(sobely)
        L_w = sobelx.T.dot(cx*phi_rW).dot(sobelx) + sobely.T.dot(cy*phi_rW).dot(sobely)
        L_f = sobelx.T.dot(phi_f).dot(sobelx) + sobely.T.dot(phi_f).dot(sobely)
        A_f = alpha_diag.T.dot(L_f).dot(alpha_diag) + gamma*phi_aux

        bW = alpha_diag.dot(phi_data).dot(J[i].reshape(-1)) + beta*L_f.dot(W.reshape(-1)) + gamma*phi_aux.dot(W.reshape(-1))
        bI = alpha_bar_diag.dot(phi_data).dot(J[i].reshape(-1))

        A = vstack([hstack([(alpha_diag**2)*phi_data + lambda_w*L_w + beta*A_f, alpha_diag*alpha_bar_diag*phi_data]), \
                         hstack([alpha_diag*alpha_bar_diag*phi_data, (alpha_bar_diag**2)*phi_data + lambda_i*L_i])]).tocsr()

        b = np.hstack([bW, bI])
        x = linalg.spsolve(A, b)
        
        Wk[i] = x[:size].reshape(m, n, p)
        Ik[i] = x[size:].reshape(m, n, p)
        # plt.subplot(3,1,1); plt.imshow(PlotImage(J[i]))
        # plt.subplot(3,1,2); plt.imshow(PlotImage(Wk[i]))
        # plt.subplot(3,1,3); plt.imshow(PlotImage(Ik[i]))
        # plt.draw()
        # plt.pause(0.001)
        print(i)
    return (Wk, Ik, W, alpha)