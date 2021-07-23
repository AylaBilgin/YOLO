from google.colab import drive    #Drive dosyaları Colab'a bağlanır.
drive.mount('/content/gdrive')

!ls
!rm -fr darknet
!git clone https://github.com/AlexeyAB/darknet/   #Darknet mimarisi darknet dosyası içerisine kurulur.
  
cp /content/gdrive/MyDrive/data_for_colab.zip /content/darknet    #Drive içerisinde bulunan data dosyası darknet dosyası içerisine kopyalanır.

!unzip /content/darknet/data_for_colab.zip    #.zip dosyası darknet içinde arşivden çıkarılır.

!apt-get update
!apt-get upgrade    #Son güncellemeler yüklenir.

!apt-get install build-essential
!apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev   #Gerekli bağımlılıklar kurulur.
!apt-get install libavcodec-dev libavformat-dev libswscale-d

!apt-get -y install cmake
!which cmake
!cmake --version    #Darknet ortamını derlemek için kullanılan make komutunu çalıştırmak için kurulum yapılır.

!apt-get install libopencv-dev
!apt-get install vim

%cd /content/darknet/
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile
!make
!wget https://pjreddie.com/media/files/tiny.weights
!chmod a+x ./darknet    #Darknet dosyasına girildikten sonra içerisinde GPU'yu çalıştırmak için gerekli düzenlemeler yapılır. Tiny.weights ağırlığı indirilir.

!ls
!../cd
!ls

!apt install g++-5
!apt install gcc-5
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 20
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 20
!update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
!update-alternatives --set cc /usr/bin/gcc
!update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
!update-alternatives --set c++ /usr/bin/g++
!apt update -qq;
!wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
!dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb    #Cuda kurulumu yapılır. Bu bölüm diğer bölümlerden daha uzun sürebilir.
!apt-get update -qq
!apt-get install cuda -y -qq    #gcc-5 g++-5 
!apt update
!apt upgrade
!apt install cuda-8.0 -y

import tensorflow as tf
device_name = tf.test.gpu_device_name()   #Kurulan Cuda'nın versiyonu belirlenir.
print(device_name)
print("'sup!'")
!/usr/local/cuda/bin/nvcc --version

%cd darknet
!make   #Darknet ortamı derlenir.

%cd darknet
!./darknet detector train data_for_colab/obj.data cfg/yolov3-tiny.cfg /content/gdrive/MyDrive/yolov3-tiny.conv.15 -dont_show    #Eğitim başlatılır.

!./darknet detect cfg/yolov3-tiny.cfg backup/yolov3-tiny_final.weights data_for_colab/data/12.jpg   #Test edilecek görsel eklenir.

def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  #plt.rcParams['figure.figsize'] = [10, 5]
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
  
  
def upload():
  from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)
def download(path):
  from google.colab import files
  files.download(path)
imShow('predictions.jpg')     #Çerçevelenmiş görsel çıktı olarak alınır.
