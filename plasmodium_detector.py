
import cv2
import numpy as np
from keras import models
from sklearn.cluster import MeanShift
from scipy import ndimage
from nms import non_max_suppression_fast
from keras import models
import tensorflow as tf

model = models.load_model("simple_cnn.h5")
model._make_predict_function()
#graph = tf.get_default_graph()

#roi generation step as shwon in Figure 6 on the paper
def ROI_generation(img,log_stage,bounding_box_scale,image_size,windows_size):
    #global graph
    
    img_gray =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,1] #select the saturation channel
    #multi-scale Laplacian of Gaussian to get the ROIs
    LoG = ndimage.gaussian_laplace(img_gray, log_stage)

    #otsu binarization
    ret, th1 = cv2.threshold(LoG,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #morphology operation opening
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th1,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    #morphology operation dilation
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    #get the centroids of the binary-image
    contours, hier = cv2.findContours(sure_bg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    moments = [cv2.moments(cnt) for cnt in contours]
    centroids = [(int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) for M in moments]
    
    
    #cluster the redundant ROIs with meanshift clustering
    if len(centroids)>1:
        centroids = np.array(centroids)
        clustering = MeanShift(bandwidth=100).fit(centroids)
        clust_centers = clustering.cluster_centers_
        
        radial = [[] for i in range(len(np.unique(clustering.labels_)))]
        
        #cluster the bounding boxes
        for i,c in enumerate(centroids):
            radial[clustering.labels_[i]].append([int(c[0] - (windows_size[0]/2)), int(c[1] - (windows_size[1]/2)),
                         int(c[0] + (windows_size[0]/2)), int(c[1] + (windows_size[1]/2))])
        
        #flexible bounding_box size
        length = []
        for r in radial:
            r = np.array(r)
            length.append([int((np.max(r[:,2])-np.min(r[:,0])) / (1/bounding_box_scale)), 
                           int( (np.max(r[:,3])-np.min(r[:,1])) / (1/bounding_box_scale))])
        
        roi = []
        for i,(x,y) in enumerate(clust_centers):
            roi.append((int(x-length[i][0]),int(y-length[i][1]),int(x+length[i][0]),int(y+length[i][1])))
    
    detected = []
    #classify the ROIs
    for (x, y, w, h) in roi:
        #unknown Assertion error
        try:
            
            c = cv2.resize(img[y:h, x:w],image_size)/255
            c = c.reshape(1,image_size[0],image_size[1],3)
            #with graph.as_default():
            p = model.predict(c)
            if p>0.9:
                detected.append((x,y,w,h))
        except Exception as e:
            print(e)
            pass
    return detected


def generatePrediction(img,save_dir): 
    windows_size = (150,150)
    initial_bounding_box_scaler = 0.25
    max_bounding_box_scaler = 0.4
    step_size = 0.05
    image_size = (100,100)
    #input image
    #img = cv2.imread(d)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    bounding_box_scales = np.arange(initial_bounding_box_scaler, max_bounding_box_scaler, step_size) #list of bounding box scales
    log_stages = 3 #LoG stages
    
    #shown in pseudocode 1 on the paper
    ROI_candidates = [] 
    
    for scales in bounding_box_scales:
        for r in ROI_generation(img,log_stages,scales,image_size,windows_size):
            ROI_candidates.append(r)
    
    
    #convert image to BGR to evaluate the result
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)   
    
    #merge redundant bounding box using non_max_supression
    roi = non_max_suppression_fast(np.array(ROI_candidates),0.3)
    
    #draw all bounding_box
    for (x, y, w, h) in roi:
        cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)
    
    cv2.imwrite(save_dir,img)
    


"""
d = "C:\\Users\\lenovo\\Documents\\kuliah s2\\computer vision\\Deteksi Parasit-20190418T085731Z-001\\Deteksi Parasit\\Data\\1.png"
save_dir = "static/uploads/"

generatePrediction(d,save_dir)
"""

