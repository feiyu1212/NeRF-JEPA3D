import cv2
import numpy as np

def concatenate_images_horizontally(image_list):
    # 计算拼接后图像的总宽度
    total_width = len(image_list) * 128

    # 创建一个空白画布，用于拼接图像
    concatenated_image = np.zeros((128, total_width, 3), dtype=np.uint8)

    # 在画布上拼接图像
    x_offset = 0
    for image in image_list:
        concatenated_image[:, x_offset:x_offset+128, :] = image
        x_offset += 128

    return concatenated_image

# 假设你有N个图像的文件路径列表，存储在image_paths中
# 示例：
image_paths = ["image1.png", "image2.png", "image3.png", ...]

# 读取所有图像并调整大小为(128, 128)
images = [cv2.imread(path) for path in image_paths]
images_resized = [cv2.resize(image, (128, 128)) for image in images]

# 将图像拼接成一行
concatenated_image = concatenate_images_horizontally(images_resized)

# 显示拼接后的图像
cv2.imshow("Concatenated Image", concatenated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



import imageio
import os
import cv2
import numpy as np


urls = [
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs/000008.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs/000030.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs/000042.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs/000292.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs/000371.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs/000476.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs/000480.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs/000687.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs2/000146.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs2/000167.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs2/000277.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs2/000350.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs2/000489.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs2/000515.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs2/000530.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs2/000534.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs2/000600.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/car_gifs2/000664.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs/000014.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs/000207.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs/000229.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs/000281.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs/000411.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs/000860.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs/000868.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs/001002.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs2/000141.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs2/000153.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs2/000342.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs2/000496.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs2/000517.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs2/000837.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs2/000931.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs2/000955.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs2/001144.gif',
    'https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/assets/chair_gifs2/001294.gif']


for gif_url in urls:
    gif_reader = imageio.get_reader(gif_url)
    frames = [frame for frame in gif_reader]
    # Convert the frames to numpy arrays (optional)
    frames = [np.array(frame) for frame in frames]
    folder = gif_url.split('/')[-1].split('.')[0]
    os.mkdir(folder)
    for i, frame in enumerate(frames):
        cv2.cvtColor(cv2.imread(f)[:, :128], cv2.COLOR_BGR2RGB)
        cv2.imwrite('{}/{:06d}.png'.format(folder, i), frame)


import glob
fs = glob.glob('./*/*.png')
for f in fs:
    im = cv2.imread(f)
    im[:20, :] = 255
    cv2.imwrite(f, im)





/data2/nerf_data/chairs_test
frame = 





import glob
import numpy as np
import cv2

targets = [(f, cv2.imread(f)[:,:]) for f in glob.glob('/data2/nerf_data/chairs_test/*/rgb/000064.png')]
srcs = [(f, cv2.imread(f)[:, :128]) for f in glob.glob('/data2/nerf_data/srn_result_gif/chairs/*/000064.png')]

d = {}
for f1, m1 in srcs:
    m1[m1 > 200] = 0
    d[f1] = []
    for f2, m2 in targets:
        m2[m2 > 200] = 0
        if m2.mean() < 6:
            continue
        d[f1].append((f2, np.abs(m1 - m2).mean()))

s = set()
for k, v in d.items():
    print(k, sorted(v, key=lambda x:x[1])[0])
    s.add(sorted(v, key=lambda x:x[1])[0][0])



        if np.abs(m1 - m2).mean() < 10:
            print(f1, f2)
           



            break
    # print(f'finish {f1}')



m1 = cv2.imread('srn_result/000411/000063.png')[:, :128]
m2 = cv2.imread('/data2/nerf_data/chairs_test/5bc916f8b9d0a7c6b40f0ac0fb9a650d/rgb/000063.png')[:, :128]

 























response = requests.get(gif_url)
gif_content = BytesIO(response.content)
# Initialize the list to hold each frame
frames = []

# Read the GIF and extract each frame
gif = cv2.imdecode(np.frombuffer(gif_content.read(), np.uint8), cv2.IMREAD_UNCHANGED)

while True:
    ret, frame = gif.read()
    if not ret:
        break
    # Convert the frame to RGB format (OpenCV uses BGR by default)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

# Release the VideoCapture object
gif.release()







nohup ls cars_test/*/rgb/000064.png | xargs md5sum > car_md5.text &
nohup ls chairs_test/*/rgb/000063.png | xargs md5sum > chairs_md5.text &

 

numpy.array_equal()
































import imageio
import os
import cv2
import numpy as np


urls = [
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/95d01543b46b5e43f398d0c5832df00e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/96a3c3653c343db6ba8a1820ecdbe891.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/96ff36cb731b29a61ad88f716ea80910.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/97bf4aac2d956c1d5d9e9a1d5cade7db.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/982f52d3117411a37ec3f7c14c03a92c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/98ad42e08a991125f0ea0ee719f6dcd1.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/98b920157670bcdd716d882f857922cf.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/98cff6064749a5f3e746404352385716.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/99f296d0bbae5585414ff38ecabd5968.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/9a8a760dd2094921bb476b1cb791329b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/9c8dd83df9678d3bc33323f64a5f289e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/9d29c69103a67b3bcb83b686440a3f01.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/9ea3e05166af97ed20363e2561dd589a.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/9ec130a3ef44b7a1e47833b310955a9f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/9f7eb24e82020dcf40df330a2f73b17c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/a05dda1538ddcb4cd747b49524a1246e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/a1d1f232168607c81dd4da6e97c175c2.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/a3f14846404245d5bbbcb091dd094e5f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/a551777c3b35c06f43f491729931f44.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/a9957cf39fdd61fc612f7163ca95602.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/a99cc4f1326b9b48b08cd968d6703683.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/ab5c8c38528f06daf398d0c5832df00e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/ab651261126de20c145adb610a878e88.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/ad2e92448857e1cc6ad08aa387990063.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/add914368a6ca448732bda87f2718525.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/ae4bcb4baca763c24521562865ab775e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/aed2ee05cf37c85c9a8c31231dd99d82.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/aed97e60cd2802ce7ffb47acd56f396b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/afe96f3cf256cbac81a6b6721af23c58.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03691459/b0ba01e0a7f3553897950841baebd2bd.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2b236f0333fe789ed4b0a2774191078f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2b7101c3a9c9aea533116515e458cb86.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2b711c735df95aded8df13fb69e08d76.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2b72fb5d7ce07564f961db66e477976d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2b766a7abde647c7175d829215e53daf.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2b8c1b23b617671d1a964dea8a7aab.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2b92417cee9dfe0cb94d1aa50e685ffa.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2ba7ea78b9b6e2a6b420256a03f575c0.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2bcdaa6516a98b552abd245afcc79403.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2be8bd6b150a9fee97df6383472cc5b6.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2bf9ed80cb75411f58dbdf2b5c6acfca.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2c6b14bcd5a5546d6a2992e9465c023b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2ca59da312d06381b927782fc69a1fbb.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2cbe6850de4fa5a34a6a7ab072f22d7d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2cc4573bb9441493d12568d09c2fba02.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2cdf160f331b2f402f732d6dce34e9b3.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2ce3965eb931e7c1efdff89bf9a96890.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2d031e07ae160bcfdd141480e2c154d3.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2d096d5dc0e0eeb370a43c2d978e502e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2d1718bda87f59dc673ddeabdcc8c6e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2d1adb247cc7a1e353da660a6567c5ff.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2d5c34ee6afae94a23c74c52752d514f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2d5c6f88dc81b668283ffcfc40c29975.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2d730665526ee755a134736201a79843.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2dbd3e61fe27cd29bddac102ee6384.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2dbfc333297696121ca2e8373597f711.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2dbfe9c041fc6b3a94ce9d7b04676231.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2dc8ffef43f5b6f6d7244a827bab8749.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2dcc5ed5780462da124dae0a32996c4c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02958343/2dd174957a2053345fcb05674109534a.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4bb41171f7e6505bc32f927674bfca67.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4c5b8babafbb7b5f937ae00fead8910d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4d0898c7cac1199a4b0853084d4042f7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4d2d87e61a18760ff64801ad2940cdd5.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4de5861211ad5b95e7ef9fff09638f8e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4e4ae13865bf47f41adbb4c4e06ad649.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4e67529b0ca7bd4fb3f2b01de37c0b29.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4ea714b785e24f0d9a6e43b878d5b335.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4eced94670d10b35e856faf938562bd0.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4ee48907120e261c3df16180af9974ee.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4f2830d3d566220be5dd38f670a033a8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4f3a64164fbe16f54c2c88971423d0be.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4f3f39ddde5874f2db73445864b72015.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4fb10ce02887e35efca1f1143bb6bc17.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4fd9c86e43a1dea17209009cfb89d4bd.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/4fe076aa34c706b83d7edb3bb2d24b58.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/50755e616df58fe566cf1b4a8fc3914e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/52185f504ffa9b32ca8607f540cc62ba.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/525446bc8f55e629151f2df200a24ac.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/52712e1c07ea494419ba010ddb4974fe.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/52747021197c7eeeb1a0518c224975f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/52a1b6e8177805cf53a728ba6e36dfae.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/52a84fea7c314f4c3dfc741b4df74043.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/52c9b1a9f8d3cbcb9a6e43b878d5b335.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/52cd5876945106d154eefcdc602d4520.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/52e7f93d592622a9615ba7bf3620290d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/530540dc968b396d7f3805c1aec66f00.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/53edcc6832e776dcca8607f540cc62ba.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/53f0e2f6671346ae5ff3feb917a6004b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02691156/543412ccea0db2f8f37f38dedb2f1219.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/e7b9c172059d23a6f12f3a2695789ca4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/e912fab93eb63fef6b4897a481d7d16a.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/e9466e872848075d3aeab48ed64c0fa4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/ea4b90cca359e2d38b9be827bf6fc77.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/ea7dc70f0ef4b04bcbe0b8af4aec5d6c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/ecb3d57cc0e8018f3f6b4923416758fd.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/ecb96d46eca5724fef0411fcb48f0299.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/ed31aec87c045ebdebe17c8dfb911d2c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/f1a3e41b45de104a810988cb5fefedde.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/f1d77e0f4a2adc2bb305a938e0ed1b48.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/f240248beae8d20661049a5d9182333f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/f3d4cb310a6106f5e66202687a227eab.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/f480ab303b2b595eb9f9e1b0ade9a295.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/f4877a34163978f84efc7e7114f1b9c5.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/f6515d1343e25bab8913de0e5cfdcafb.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/f800fbcdef1ac9038b5dbcd4be5ceef8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/f84f6c14852ddad1d06e6be4f8954ac.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/fc314f1e3325a47af287ec53a469521.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/fc542f42b786ae20c40162916d18053d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03211117/fdf3953c665c36fdeb47c06d7d8c2d65.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/11126129151f3e36afb1ffba52bacfa2.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/12038871e583f645af56497f051566ab.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/1345ba2c3b27ba786bb9681d4604e123.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/138cbc9b5456cfef55d33831e71fa52.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/13ff0ba1e8d79a927f71da36e5c530dc.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/14139593048f806e79093d8935cfe4f0.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/154d2a83fc91d473d735fa1ab17311ec.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/1632f8ce3254cfccc8c51c73cb7275ed.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/164248eefde5ce846314c3b6a3a65519.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/739971469c9903eeb6dc6c452bb50aac.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/73ce9a5617acc6dbf1e0fcef68407ae5.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/73e0ab2e1d0ea752bc6e84bc30e573cd.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/74a38b74bb57a2af6a405aa597e7fe24.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/74e930c54ddaf8add34ad43a472bf958.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7537fb364b25e32e999562d1784e5908.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/76377dc3d6b1dad2c0aaedf10d776af7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/76d55781aaac09f918b61860f81c82f7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/76dd6c8a2cf09887bbb7f70943d3cb52.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/77241daf76a045c099d9d900afe054b8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7744efae453f26c05e9263096a26104d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7787bf25c9417c4c31f639996cb3d35d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7816689726ed8bbfc92b24247435700c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/78a0c4ff75258ecf16b34c3751bc447d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/78dd4dc65639258cd735fa1ab17311ec.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/78e2a75ff1d93138e8914057d776d90b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/79461905bb97ebc21a77229b9f90bf5.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/79f507c3befe69ba2987a7b722c00b7d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7a79d63c2cf6df519f605c8c86eb1ec2.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7aff2693a24b4faed957eaf7f4edb205.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7b1b02b8f71226e4bb9224760a70fece.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7bd6db14ec6c37efeac2c9f41d1276bf.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7bfdd659a04c412efa9286f039319ff7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7c31ae88ca4faa649a2ee232a197081e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7c426a52358e8c1d64c4db7c3b428292.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7c6a21d8b91fd12a1b7837f7a64e3031.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7d310ff81ee66564b38e8b1e877a5704.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7dba6294173994131226a6f096e4f8c8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7e316474bc8b072fca74c4e4ab012aef.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04090263/7e5c0d35215be21ff3998727b15249db.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/52d307203aefd6bf366971e8a2cdf120.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/52dd0fac460adb45e2879d5d9f05633.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/536cae63d37ef32265ba78ad9601cf1b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/541e331334c95e5a3d2617f9171b5ccb.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/54a209955f7a47bed8e8a8a207ee5bd2.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5560a425c597eacbff841fc99bb16039.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/55e0dfba8cc226871b17743c18fb63dc.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/55f6500b52310f26352ecf815a233abb.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5649e603e8a9b2c295c539fc7d92aba.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5660a383172900a6593ebeeedbff73b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/56652a99cd43b77744dace04559bf008.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/569c7293b52b633814038d588fd1342f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/56cafcac4df5308d35dda488a4bbb1e1.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/572da8680677fe8937b2bb75885cfc44.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/57ceb2601802a37e534fa06200d07790.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/57f5a7660b1f186e14038d588fd1342f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/580e58ca5b0f8dcf490ad276cd2af3a4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/58447a958c4af154942bb07caacf4df3.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/592fdb44d4bbf0ee490ad276cd2af3a4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/59c32d74e6de63643d41bddf307a46a8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5a94cc0c277f391df9aec59741c69cf7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5b23328fa6d5382d295b24579cf55b8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5b5bd4ca75b788c6ece5b3f5f7505a42.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5bf5096583e15c0080741efeb2454ffb.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5cc1b0be0eb9a6085dca6305fb9f97ca.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5d2ff1a62e9d58baeaf288f952624966.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5d9f1c6f9ce9333994c6d0877753424f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5e6fe3ce7afd4490441e918a50adf356.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5e9e4d6551b775d8cabc196a7f35f1aa.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04256520/5eae999a81861da03bec56bff764ba78.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/7a8615c643bc3d96ed6eef8e856a36ea.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/7b39100755e9578799284d844aba7576.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/7bc1b202ebf000625949e084b65603cf.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/7be01530bf43f2ed8a83637b92bdc7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/7bebdd742342ba93febad4f49b26ec52.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/7c23362b39f318cbb18d6f615cb18bdd.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/7eadde33d9f9d8b272e526c4f21dfca4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/8006e3cb3ba934a05b977412e02c412c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/802c287eacfbcba9564b8e0b814c602e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/80436dff2a30721849655ac7c771b113.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/809ed642119aa5799e8bf807e902261.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/80a5ce76688700a2fdd36147bc6b8031.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/81bbe86e17196c2fd0db1e44373ee2d4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/8508808961d5a0b2b1f2a89349f43b2.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/8522fb13bfa443d33cabd62faf4bd0f0.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/85574f6036ea1f90d8c46a3a266762d7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/85a73c46a97649fa6d0c88a73d7cb14d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/85f8a8c585742c9b96a3517f50eeb9f4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/86d556273aa5075aaa660c42e675c161.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/87107afb7ad115414b3c42e318f3affc.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/873aad4a222d541a91c2792fcdc1ca8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/882aae761370df14786810c22b062a88.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/886ff4f5cd90c6ad39b3360f500ac52a.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/8872dceb7ba9f34c140406de8e63ea3a.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/892900c66731d9c473ab7b7128d466a2.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/8937a2d361775c68aafd61baec633e88.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/896abd405c79547086485c798787f66b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/89ed63af13b79f3ef42a90fe4baf4591.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/8a6944062cbf8b25ef0a7c6e0ed55209.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03636649/8a9f2e5b726ea37f60ad823977adaa23.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/26ab5349a902d570d42b9650f19dd425.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/272a4cf3cfff3eb1e173cee47fbaa88.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/27805445a3c587c1db039d8689a74349.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/279351131d06222cbe9bca6d7b2e5b3.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/27a90972dfe64be5c3bd24f986301745.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/27f9965a337bf46d85924458b86f34.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/28001cb70c38f19cf32b6091d9628440.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/281f296380a0e4a81db7abc68608fde1.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/282d36d7ca9705f6ca421e9e01647b4a.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/285857e7d12f1b74a4d2a71d4ca57f99.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/28912046d42946df7db48c721db3fba4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/28e64eefcada205fddf120185fc0b2a0.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/28f3844a7a3d37399c0ca344f487323e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/28f702b5c6ccffe7fcf9154182ccb5a4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/28fb9a81898f88c4ae8375def5e736d8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/29207ae4e01c415894fc399eb2c2f348.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2927b65bc7b9561bf51f77a6d7299806.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/29b55c6cc05c37606e066e11deddab75.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/29c6a184dfee3050820018801b237b3d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/29def96b77d4cfe372f9a9353d57f9ef.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2a0eff86efdf60a89a682a6d0e92c30.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2a896f1d6925cc61dc89a28b5e6678e0.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2aad9a8f3db3272b916f241993360686.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2ab09f4db5f5e842bf595b60a303303.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2ab79a94145330a95ca21a5844017a0f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2abe61af67cbd99aaa1d46a2befc5e09.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2ae89daf7433f4d14b3c42e318f3affc.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2b06a917abc1150b554ad4a156f6b68.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2b0c16b26ebfb88f490ad276cd2af3a4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04379243/2b1c1e0fad3cb3b8fad46760e869d184.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9004946f75082a8632c0857fb4bcf47a.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/90bf73b91185303139555c8c231d0eb7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/90d83e1dde32426407e66c6e74f5ce3.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/90e6c6083fcd47833e45dd2f173cbf9b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9115745cde6f30c67f141c9abbacbdb8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/91a124454518abb7f2ad837508eb2db7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/91e0e1a6dbf302c3d55da98ad008849b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/921a5d88994aa99c71327f667b2179b0.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/925c05dbefada808cfe472915a175bb.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9262aa413df7b369d735fa1ab17311ec.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/92e4ae4dfff684832dbef90d406185fa.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9380065fd20eec791a7be2887bc37d1.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/94ddf20a9a6e035e85f7a3de54751f1b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/94e1c24d7f205a39779adafea0303e70.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/94e216dc57731577c14e2939682bc455.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/950ebca8ad7d94051fba2cab1ada6bf6.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/954c459bc6762abc24f2ecb72410a6d9.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/956c3b989bdd0603158a3417d0510bc.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/969163f17ce6467d9378da473ee38a8d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/98cbe2b3e62879058e3175d49fbb0f30.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9951a6732eb8438a79662f01dd94fba1.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/996c90952cfff5b24baa0720a34ff704.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9b90b9cbd9577d842b72b4a851b36ab9.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9b93b845578ee8a20c10ff7bcef26d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9c2c87ceba2a465f86b234e3f0128df2.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9c50b10fb651e57fdd93d77eaf89012.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9e3c0b7fb69ec3997cd1f8dd6fbce8fb.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9e49192ba54ab0fc92c108c58096cae.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9efd4dac9e4b1698876eb99526752ffb.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04530566/9f468767b1fd9285eb2c303a0e0d287b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/895563d304772f50ad5067eac75a07f7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/89e2eaeb437cd42f85e40cb3507a0145.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/8a5a59ab999c03ccfb0eb7e753c06942.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/8a6f07f3d357fbfd2b12aa6a0f050b3.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/8aabc6c97aeacae7ad5067eac75a07f7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/8b98dbc11c5d2fb7601104cd2d998272.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/8d1f361eb7a927d8907921e9162f6a43.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/8d218bceb517f272f155d75bbf62b80.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/8f52743c3cb564f149e6f7df978f3373.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/8f64075972a1f7f3dc18af6f6bfce3ef.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/9027bc9728f006bb40f0ac0fb9a650d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/90a309f7ac2c947f155d75bbf62b80.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/90a8b80aa2fa209ca936e2693dce34b6.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/916a6ff06a128851ed98cca8f0ccd5f7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/919f90e92283eff8febad4f49b26ec52.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/91e169ea3ceb587beff42b9e13c388bc.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/92f1fa8d3b5da497ad5067eac75a07f7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/943dde2754ddc7822e8ff3556a90169.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/95b375118d800de7ad5067eac75a07f7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/95eed587c3728d22601104cd2d998272.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/9699995246fd521ca909cd1ba5751669.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/976636850849c3d6ffd996335233167.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/9787c8521ba8d46b5b83c5170da4c6c2.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/9790980a8ff823287ed9296ee19fa384.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/98971eb747738a6880360680c1602c7d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/9897a75c64a369d458c73770090b865.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/990c56d6ab64279c2056b4bd5d870b47.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/991803aca7fca258b40f0ac0fb9a650d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/9ad5cab6ff1e45fd48113d3612de043b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02828884/9bd9483c8eeeb703cb2a965e75be701c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/93ee050394f3b15e3c8d0fdfb1cc2535.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/941289c22ad19099a87002a4eeaf610.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/948709ebfb6cd6ff6f2222642bd41c09.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/94d10abadfa0d88bf51f77a6d7299806.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/94f4f2342f6335c5875c4d98e634f167.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/951377627e2fb20f86d53ab0fe94e911.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9524af09747661fbe0f91080da8431d7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/956c437be87a2be5f51f77a6d7299806.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/962b62d2b3823aff51f77a6d7299806.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/96969d7adc540bf686d53ab0fe94e911.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/97b415cd78587de5fa29682ba98e856d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/98d963a9f353cd026b0f9e3b3feb2454.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9990d6c43d95c521c4df475ad1a6d009.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/99ff3359d64f1f45ce5d6e9371bb5c33.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9a195ea2a21bc7511a4db721d603d852.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9b28e4f5c66daae45f29768b6bb620cb.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9bcfb450ed3046a74da5feafe6f1c8fc.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9c802e8971c1b25f480fa521a9c7198a.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9cec9c8b65ec9e4f4c62bee40dcdc539.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9dfac0132596ff09b13b0af4c7a59aa0.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9e6434ba1ad59fa611fc2b865c2a185b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9f07ee5bcf6a4b11151b305e5243d9f8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9f17f45a28d063e7391e4d6c585a697a.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/9fe8af4bc8683818579c9a60a6b30a2a.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/a03797c4034be11ac59350d819542ec7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/a07f3b931238a5c3fe618e12c7f65698.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/a08aa6e386cd983c59350d819542ec7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/a0beaab9a8c2f0954e7d60def15dcb8b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/a0eb46b125a99e26473aef508bd8614e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/02933112/a3a6f9e80956ec6f4035f93ab9531db.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/eed72e5bc3dc4e85150c05906b260c9e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/f18dbf3cbc7d3822de764ca2f457c756.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/f46531484dea3574a803a040655859ad.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/f649a5368586aa77eecb4e0df212ad9.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/f75cac5b112f14002c32dcd0becbedb7.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/f77811bd35b9215dfd06b6d6f44d62dc.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/f7cb2d921f246c00cc8eea03ea8843cc.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/f928f74ed34e46c4b5ce02cb8ffbdc86.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/fa14355e4b5455d870f7cd075189ddd8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/fbd120d2c01484d56c95c6d882af3c0.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/fe39a57b8137ecbd5b2233351507f22f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/04401088/ff1e484e02e001bdf8a0d9b8943b4050.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/108238b535eb293cd79b19c7c4f0e293.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/108b9cb292fd811cf51f77a6d7299806.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/10dc303144fe5d668d1b9a1d97e2846.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/11040f463a3895019fb4103277a6b93.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/111cb08c8121b8411749672386e0b711.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/11347c7e8bc5881775907ca70d2973a4.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/114f72b38dcabdf0823f29d871e57676.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/11740d372308f12185047f9f654ddc2e.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/117bd6da01905949a81116f5456ee312.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/1190af00b6c86c99c3bd24f986301745.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/11c9c57efad0b5ec297936c81e7f6629.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/11d4f2a09184ec972b9f810ad7f5cbd2.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/11e28120789c20abc8687ff9b0b4e4ac.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/11e6e58798ae5be83e5b5dbd84cdd0f8.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/124117cdec71699850c2ec40da48fd9d.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/128517f2992b6fb92057e1ae1c4cc928.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/12f395270a3316d01666e1246e760f82.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/326a0c5116e410cfe6c1a071a4e8216c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/326f74afbed5d727da8b0c70313fbbae.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/3276361e9e651238ac4ed5495364a497.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/3289bcc9bf8f5dab48d8ff57878739ca.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/328df096e089c4eafebad4f49b26ec52.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/329c2234d134dc89492d9da2668ec34c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/329ec98f10af7214ac6962daa1b6ab91.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/32a4ddf426cef33c323ad87fe7d4deee.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/32a9329c11b5c35d4b3c42e318f3affc.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/32f2998a16e477163c4f66791e25960f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/32f918efaa64a4d9c423490470c47d79.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/3358536e8e7c416ea9ef8e11754eeede.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/337050c0a5a7f965cc5cf3ad66086732.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/33aaad494817a6f4ab705559ec99536f.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/33e436e30da86f3bc5beea20858a99d5.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/341a2d3df9800314fa260f4362cac599.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/3457146f11dab7eff3c9c1464e55d580.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/348528e8d474a003cb481b0b11df1849.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/34898c36e711fbde713decb1a0563b12.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/34b1b2ee54ea314486a1cb4b695edbd9.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/35053caa62eea36c116cc4e115d5fd2.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/3526528270d5f3f766268502c798b9af.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/357275196334dc4feaf288f952624966.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/359331da71ed26eca6c03a53cf0a14c9.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/35bcb52fea44850bb97ad864945165a1.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/35c2de57ee36f652492d9da2668ec34c.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/35e77eed59e1113c22e4620665c23c97.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/35ee4bcad88ab50af6e44a01c524295b.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/35f83268d4280532dc89a28b5e6678e0.gif',
'https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/panel_gif/03001627/367dc1e6752cabbcc34bba142e6e15e6.gif',
]


for gif_url in urls:
    try:
        gif_reader = imageio.get_reader(gif_url)
        frames = [frame for frame in gif_reader]
        # Convert the frames to numpy arrays (optional)
        frames = [np.array(frame) for frame in frames]
        folder = '_'.join(gif_url[:-4].split('/')[-2:])
        os.mkdir(folder)
        for i, frame in enumerate(frames):
            _ = cv2.imwrite('{}/{:06d}.png'.format(folder, i), frame)
    except:
        print(gif_url)















