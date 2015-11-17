int *getMatrix2(IplImage *image, int channel);
void printMatrix(int* matrix, int height, int width);
void conv(int*, int*, int, int);
void printMatrix(int *mat, int rows, int columns);
IplImage *loadImage(char *path);
void showImageProperties(IplImage *image);
int *getMatrix(IplImage *image, int start, int end, int channel);
int *emptyMatrix(IplImage *image, int stride_size);
IplImage *convolution(IplImage *image, int stride_size);


