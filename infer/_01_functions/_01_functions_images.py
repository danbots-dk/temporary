from _00_configuration._00_settings import *
from _00_configuration._01_variables import *
from _01_functions._00_helper_functions import printif, printonceif
#from _01_functions._04_functions_plots import *

def histo_stretch(img, pmin = 2, pmax = 98):
    vmin,vmax = np.percentile(img, [pmin,pmax])
    rescaled_img = exposure.rescale_intensity(img, out_range=(vmin,vmax))
    return rescaled_img

def threshold_nd_close(img):
    """
    isolates the background 
    """
    threshold = max(img[0,0],
                img[0,-1],
                img[-1,0],
                img[-1,-1])
    t_img = np.where( img > threshold, 1, 0)
    close_img = morph.area_closing(t_img, area_threshold = 150)
    #print (close_img)
    close_img_2 = np.where(close_img <= 0, 1, 0)
    background = morph.area_closing(close_img_2, area_threshold = 200)
    return background

def change_background_brightness(img,delta):
    """
    isolates the background by thresholding and area_closing
    chages the brightness of the background
    stitches up brightbg and foreground into a final image
    """
    threshold = max(img[0,0],
            img[0,-1],
            img[-1,0],
            img[-1,-1])
    t_img = np.where( img > threshold, 1, 0)
    close_img = morph.area_closing(t_img, area_threshold = 150)
    #print (close_img)
    close_img_2 = np.where(close_img <= 0, 1, 0)
    mask = morph.area_closing(close_img_2, area_threshold = 200)
    notmask_log = np.logical_not(mask)

    masked_img = np.ma.array(img, mask = mask)
    masked_img_2 = np.ma.array(img, mask = notmask_log)

    input_im_bright,delta = change_brightness(img, delta = delta)
    value = input_im_bright[0,0]
    #create background with different brightness
    bg = np.full(img.shape,value)

    # Zero background where we want to overlay
    bg[masked_img_2.mask == True]=0

    # Add object to zeroed out space
    final = bg + img*(notmask_log>0)
    return final

def change_brightness(im, delta= None, deltas = np.arange(-0.4,0.4,0.25)):
    """
    takes a np array, converts it to tensor, applies a randomly chosen delta
    returns brightness adjusted image and delta
    """
    tf_img = tf.convert_to_tensor(im)
    
    if delta is None:
        delta = random.choice(deltas)
    brght_img = tf.image.adjust_brightness(tf_img, delta = delta)
    return brght_img.numpy(),delta

def string_num(num):
    """
    to help visualize sign of values in titles strings 
    """
    if num> 0:
        mark ="+"
    else:
        mark =""
    return f"{mark}{round(num,1)}"


def make_grayscale(img):
    """
    Transform color image to grayscale
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def resize( img, 
            H = 160,
            W = 160,

            plotstat = False,
            printstat = False,
            ):
    """
    Select fixed size square in the image
    TODO: is this needed? 
    """
    printif(img.shape, printstat)
    img_resized = img[0:H,0:W]
    printif(img_resized.shape, printstat)

    # if plotstat:
    #     plotListFig([img, img_resized], # list images
    #                 ["Original image", "Resized image"], # list titles
    #                 )

    # return(img)

def normalize_image(img):
    """
    Changes the input image range from (0, 255) to (0, 1)
    """
    img = img/255.0
    return img

def normalize_image_old(img):
    """
    Normalizes the input image to range (0, 1) for visualization
    """
    img = img - np.min(img)
    img = img/(np.max(img)- np.min(img))
    return img


# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

print(f"Functions Read & Edit Images import successful")
