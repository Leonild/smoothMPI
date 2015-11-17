all: clean v2

clean:
	rm -rf convolution

v2: clean
	mpicc convolution_v2.c  -I/usr/local/lib -I/usr/include/opencv -I/usr/local/include/opencv -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -Iutil -o convolution
