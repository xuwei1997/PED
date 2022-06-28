import os
from tqdm import tqdm
from PIL import Image
from unet_gm import Unet
import numpy as np
from matplotlib import pyplot as plt
from pykalman import KalmanFilter
import cv2

def Kalman1D(observations,damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

unet = Unet()
dir_origin_path = "/mnt/hdd/cherry2021/cdfg/E88569964p6_12/"
dir_mask_path = "/mnt/hdd/cherry2021/cdfg/E88569964p6_21/"
dir_save_path   = "/mnt/hdd/cherry2021/cdfg/E88569964p6_save/"

img_names = os.listdir(dir_origin_path)
img_names=sorted(img_names)
print(img_names)

img_g_names = os.listdir(dir_mask_path)
img_g_names=sorted(img_g_names)
print(img_g_names)

out = []

for img_name,img_g_name in tqdm(zip(img_names,img_g_names)):
    if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
        print(img_name)
        #img rgb
        image_path = os.path.join(dir_origin_path, img_name)
        print(image_path)
        image = Image.open(image_path)

        #img mask
        image_g_path = os.path.join(dir_mask_path, img_g_name)
        print(image_g_path)
        img_g = cv2.imread(image_g_path, 0)
        # print(img_g.shape)
        # img_g2bgr = cv2.cvtColor(img_g, cv2.COLOR_GRAY2BGR)
        #二值化
        ret, img_b = cv2.threshold(img_g, 0, 255, cv2.THRESH_OTSU)
        #滤波
        kernel = np.ones((7, 7), np.uint8)
        img_b = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel)
        img_mask = cv2.morphologyEx(img_b, cv2.MORPH_OPEN, kernel)

        # 预测
        tmp,r_image = unet.detect_image(image,img_mask)
        out.append(tmp)
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)
        r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")

out = np.array(out)
print(out)

print(out)
np.savetxt('out1.txt',out, fmt='%.04f')

# out[:, 1]=out[:, 1]-out[0, 1]
# out[:, 2]=out[:, 2]-out[0, 2]
# out[:, 4]=out[:, 4]-out[0, 4]
#
# out=np.maximum(out, 0)

plt.figure(0)
# plt.plot(out[:,0], label="_background_")
# plt.ylim(0,0.4)
# plt.plot(out[:, 0], label="background")
plt.plot(out[:, 1], label="fruit")
plt.plot(out[:, 2], label="leaf")
plt.plot(out[:, 3], label="trunk")
plt.plot(out[:, 4], label="flower")
plt.legend(loc='upper right')

plt.xlabel('Day')
plt.ylabel('Proportion')

plt.savefig('E88569964p6t09fso.png',dpi=300)
# plt.show()



out[:, 1]=out[:, 1]-out[0, 1]
out[:, 2]=out[:, 2]-out[0, 2]
out[:, 4]=out[:, 4]-out[0, 4]
out=np.maximum(out, 0)

plt.figure(1)
# plt.plot(out[:,0], label="_background_")
# plt.ylim(0,0.4)
# plt.plot(Kalman1D(out[:, 0]), label="background")
plt.plot(Kalman1D(out[:, 1]), label="fruit")
plt.plot(Kalman1D(out[:, 2]), label="leaf")
plt.plot(Kalman1D(out[:, 3]), label="trunk")
plt.plot(Kalman1D(out[:, 4]), label="flower")
plt.xlabel('Day')
plt.ylabel('Proportion')
plt.legend(loc='upper right')
plt.savefig('E88569964p6t09fs1.png',dpi=300)
# plt.show()


plt.figure(2)
# plt.plot(out[:,0], label="_background_")
plt.ylim(0,0.25)
plt.plot(Kalman1D(out[:, 1]), label="fruit")
plt.plot(Kalman1D(out[:, 2]), label="leaf")
plt.plot(Kalman1D(out[:, 3]), label="trunk")
plt.plot(Kalman1D(out[:, 4]), label="flower")
plt.xlabel('Day')
plt.ylabel('Proportion')
plt.legend(loc='upper right')
plt.savefig('E88569964p6t09fs2.png',dpi=300)
# plt.show()




out[:, 1]=Kalman1D(out[:, 1]).reshape(130)
out[:, 2]=Kalman1D(out[:, 2]).reshape(130)
out[:, 3]=Kalman1D(out[:, 3]).reshape(130)
out[:, 4]=Kalman1D(out[:, 4]).reshape(130)

print(out)
np.savetxt('out2.txt',out, fmt='%.04f')

