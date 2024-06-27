#%%
# https://stackoverflow.com/questions/49992300/how-to-show-graph-in-visual-studio-code-itself
printstat = True # True for full prints/debugging

#==================================================================================================
#imports 
#==================================================================================================
from functools import reduce
import sys
import os
from upload_point_cloud import upload_simulation_ply
from plyfile import PlyData, PlyElement
from io import StringIO


# suppress tf tons of annoying messages: 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from config import *
# from _03_Networks.not_used.NET2_WR_K_001.NET2_model import * 
# from _03_Networks.not_used.NET2_WR_K_001.NET2_parameters_combo import * 
# import random

PI = np.pi

#==================================================================================================
# print info and create directories for this training 
#==================================================================================================
date,ts = print_user_info()
# device = cudaOverview_tf()

start = 0
stop =10000
step =43



# wrapin = '/nnwrap1.png'


def main():
    inFolder = '/danbots/data2/data/wand/1-09-2023/planes/lens20/images/'
    # inFolder = '/home/samir/Desktop/pldepth3/'
    myname = 'wandfringe2'
  
    #==================================================================================================
    # Test outputs :Reload model
    #==================================================================================================

    this_dir = create_infer_nnwrap_output_dir(date,ts)

    # create_dictionary

    os.mkdir(this_dir+'/nn_depth' )
    nndepthfolder = this_dir+'/nn_depth'
    os.mkdir(this_dir+'/nn_wrap' )
    nnwrapfolder = this_dir+'/nn_wrap'
    os.mkdir(this_dir+'/sqimage' )
    sqimagefolder = this_dir + '/sqimage'
    os.mkdir(this_dir+'/ply' )
    plyfolder = this_dir+'/ply'
    wrp_folder = '/danbots/data2/data/models/nn1-1/20230901/12h54m07s/model'
    dp_folder = '/danbots/data2/data/models/nn1-wr-depth/20230816/12h27m36s/model'
    wrap_model = tf.keras.models.load_model('/danbots/data2/data/models/nn1-1/20230901/12h54m07s/model.h5', compile=False)
    depth_model = tf.keras.models.load_model('/danbots/data2/data/models/nn1-wr-depth/20230816/12h27m36s/model.h5', compile=False)
    modelID = 5

#============================================================================================

    def generate_pointcloud(rgb_file,mydepth,ply_file):
        rgb = Image.open(rgb_file)
        depth = mydepth #np.load(depth_file )
        # depth = np.fliplr(depth)
        points = []
        print(rgb.size[1],rgb.size[0])    
        for v in range(rgb.size[1]):
            for u in range(rgb.size[0]):

                color =   rgb.getpixel((v,u))

                if True: #mask[u,v]: #(mask.getpixel((v,u))>15):
                    # Z = depth.getpixel((u, v))
                    Z = depth[u,v]*1
                    Y = .23 * (v-80) *  Z/80 #.306 = tan(FOV/2) = tan(48/2)
                    X = .23 * (u-80) *  Z/80
                    points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))#(X,Y,Z,color[0],color[1],color[2]) (X,Y,Z,127,127,127)
        #file = open(ply_file,"w")
        #file.write(
        ply_text= '''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points))
        #file.close()

        with StringIO(ply_text) as f:
            plydata = PlyData.read(f)
        plydata.text = False
        plydata.byte_order = '<'
        plydata.write(f'{ply_file}')


    #=================================================================================================== #============================================================================================

    def generate_pointcloud(rgb_file,mydepth,ply_file):
        rgb = Image.open(rgb_file)
        depth = mydepth #np.load(depth_file )
        # depth = np.fliplr(depth)
        points = []
        print(rgb.size[1],rgb.size[0])    
        for v in range(rgb.size[1]):
            for u in range(rgb.size[0]):

                color =   rgb.getpixel((v,u))

                if True: #mask[u,v]: #(mask.getpixel((v,u))>15):
                    # Z = depth.getpixel((u, v))
                    Z = depth[u,v]*1
                    Y = .23 * (v-80) *  Z/80 #.306 = tan(FOV/2) = tan(48/2)
                    X = .23 * (u-80) *  Z/80
                    points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))#(X,Y,Z,color[0],color[1],color[2]) (X,Y,Z,127,127,127)
        #file = open(ply_file,"w")
        #file.write(
        ply_text= '''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points))
        #file.close()

        with StringIO(ply_text) as f:
            plydata = PlyData.read(f)
        plydata.text = False
        plydata.byte_order = '<'
        plydata.write(f'{ply_file}')


    #===================================================================================================        
    def gammacorr(image, gamma):
        imout = 255*(image/255)**(1/gamma)
        return(imout)



    #===================================================================================================        

    def siminfer():
        flist = []
        for i in range(start,stop,step):
            print(i)

            inpfile = inFolder+'/render'  + str(i)+ '/wimage.png'
            inpfile2 = inFolder+'/render'  + str(i)+ '/image8.png'
            # inpfile = inFolder  + str(i)+ '.png'                  # Choose one or the other !!!!
            sqimage = cv2.imread(inpfile).astype(np.float32)
            img = make_grayscale(sqimage)
            img = gammacorr(img, 1**(-1))
            sqimagepath= sqimagefolder +'/'+str(i)+ '.png'
            inp_img = img/255

            wrapInput = wrap_model.predict(np.array([np.expand_dims(inp_img, -1)]))
            wrapInput = wrapInput.squeeze()
            wr_save = inFolder+'/render'  + str(i)+ '/nnwrap.png'
            wrappath= nnwrapfolder +'/'+str(i)+ '.png'
            cv2.imwrite(nnwrapfolder +'/'+str(i)+ '.png', (256*wrapInput))

            depthInput = depth_model.predict(np.array([np.expand_dims(wrapInput, -1)]))
            nndepth = depthInput.squeeze()
            wr_save = inFolder+'/render'  + str(i)+ '/nndepth.npy'
            cv2.imwrite(nndepthfolder +'/'+str(i)+ '.png', (20*nndepth))
            np.save(wr_save, 256*nndepth, allow_pickle=False)
            cv2.imwrite(sqimagefolder +'/'+str(i)+ '.png', (img))
            generate_pointcloud(inpfile2,256*nndepth,plyfolder +'/'+str(i)+ '.ply')  #divide nndepth by 2 (128 instead of 256 multiply)

            plypath= plyfolder +'/'+str(i)+ '.ply'
            flist.append({
              'point_cloud_file': plypath,
              'fringe_image': sqimagepath,
              'wrap_image': wrappath,
              'name': myname,
              'model': modelID,
            })
        responses = upload_simulation_ply(files=flist, batch_name=myname, model=modelID, depth_model=dp_folder, wrap_model=wrp_folder)

        print(responses)

    siminfer()
    

    
if __name__ == '__main__':
    
       
    main ()


