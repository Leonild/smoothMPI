/* 
Command syntax:
$convolution <input_image> <output_image>
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include <highgui.h>
#include <mpi.h>
#include "convolution_v2.h"

#define MASTER 0

int maskSize = 15;

void conv(int *inImage, int *outImage, int rows, int columns) 
{
	int i, j, a, b;
	int soma = 0;
	int divisor = maskSize * maskSize;
	int padding = (int) maskSize / 2;
	int initPadding = padding * (-1);
	int endPadding = padding;

//	omp_set_nested(1);
//	omp_set_num_threads(4);
//	#pragma omp parallel for private(i,j,soma,a,b) schedule(static)
	for(i=0; i<rows; i++)
	{
		for(j=0; j<columns; j++)
		{
			if( (i<(rows-padding)) && (j<(columns-padding)) && (i>(padding-1)) && (j>(padding-1)))
			{
				soma = 0;
				for(a=initPadding; a<=endPadding; a++)
				{
					for(b=initPadding; b<=endPadding; b++)
					{
                                		soma  += inImage[(j+a) + (i+b) * columns];
					}	
				}
			}
			outImage[j + i * columns] = soma/divisor;	
		}
	}
}


int main(int argc, char *argv[]) 
{
	char *inputfile  = argv[1];
	char *outputfile = argv[2];

	IplImage *input, *output;

        int *b_inMatrix, *b_outMatrix;
        int *g_inMatrix, *g_outMatrix;
        int *r_inMatrix, *r_outMatrix;

	int stride_size, remaining, width;
	
	clock_t time;

	double time_taken;

        int numprocs, myid, proc, workers, length;

	int i, j, x;                                     // Counters

        int start, end;
        int padding = (int) maskSize / 2;

	MPI_Status status;

	time = clock();
	
        MPI_Init(&argc,&argv);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	workers = numprocs - 1;

	if( myid == MASTER ) 
	{
		input  = cvLoadImage(inputfile, CV_LOAD_IMAGE_COLOR);

		showImageProperties(input);
	
		b_inMatrix = getMatrix(input, 0, input->height, 0);
		g_inMatrix = getMatrix(input, 0, input->height, 1);
		r_inMatrix = getMatrix(input, 0, input->height, 2);
		
		int height  = input->height;
        	int padding = (int) maskSize / 2;

        	stride_size = input->height / workers;
        	remaining   = input->height % workers;
		width       = input->width;

	//	cvReleaseImage(&input);

	        for( proc = 0; proc < workers; proc++ ) 
        	{
                	start = proc * stride_size - padding;
                	end   = (proc * stride_size) + (stride_size-1) + padding;
        
       		        if( proc == 0 )
                	{
                        	start = proc * stride_size;
                	}

                	if( proc == (numprocs-2) )
                	{
                        	end = (proc * stride_size) + (stride_size-1) + remaining;
			}

			MPI_Send(&start, 1, MPI_INT, (proc+1), 0, MPI_COMM_WORLD);
			MPI_Send(&end,   1, MPI_INT, (proc+1), 0, MPI_COMM_WORLD);
			MPI_Send(&width, 1, MPI_INT, (proc+1), 0, MPI_COMM_WORLD);

			length = (end-start+1) * width;

			MPI_Send(&b_inMatrix[start*width], length, MPI_INT, (proc+1), 0, MPI_COMM_WORLD);
			MPI_Send(&g_inMatrix[start*width], length, MPI_INT, (proc+1), 0, MPI_COMM_WORLD);
			MPI_Send(&r_inMatrix[start*width], length, MPI_INT, (proc+1), 0, MPI_COMM_WORLD);
        	}      

		/*free(b_inMatrix);
		free(g_inMatrix);
		free(r_inMatrix);*/

		output = cvCreateImage(cvSize(input->width, input->height), IPL_DEPTH_8U, 3);

		b_outMatrix = (int*) malloc(sizeof(int) * input->height * input->width);
		g_outMatrix = (int*) malloc(sizeof(int) * input->height * input->width);
		r_outMatrix = (int*) malloc(sizeof(int) * input->height * input->width);
		
		for( proc = 0; proc < workers; proc++ )
                {
                        start = proc * stride_size - padding;
                        end   = (proc * stride_size) + (stride_size-1) + padding;

                        if( proc == 0 )
                        {
                                start = proc * stride_size;
                        }

                        if( proc == (numprocs-2) )
                        {
                                end = (proc * stride_size) + (stride_size-1) + remaining;
                        }

			length = (end-start+1) * width;


			//Provavelmente aqui esta o erro, pois devo comecar em start+1
                        MPI_Recv(&b_outMatrix[start*width], length, MPI_INT, (proc+1), 0, MPI_COMM_WORLD, &status);
                        MPI_Recv(&g_outMatrix[start*width], length, MPI_INT, (proc+1), 0, MPI_COMM_WORLD, &status);
                        MPI_Recv(&r_outMatrix[start*width], length, MPI_INT, (proc+1), 0, MPI_COMM_WORLD, &status);

                        if(proc == 0)
                        {
                                x = 0;
                                end = end + padding;
                        } else {
                                x = start + padding;
                        }


                        for( ; (start+padding) <= (end-padding+1); x++, start++)
                                for( j = 0; j < input->width; j++ )
                                        cvSet2D(output, x, j, cvScalar(b_outMatrix[x * input->width + j], g_outMatrix[x * input->width + j], r_outMatrix[x * input->width + j], 0) );
		} 

		cvSaveImage(outputfile, output, 0);
	
		time = clock() - time;

		time_taken = ((double)time)/CLOCKS_PER_SEC;
		printf("convolution took %f seconds to execute \n", time_taken);

                //free(b_inMatrix); free(g_inMatrix); free(r_inMatrix);
                //free(b_outMatrix); free(g_outMatrix); free(r_outMatrix);

		//if(!output) cvReleaseImage(&output);

	} else {

		MPI_Recv(&start, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&end,   1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&width, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);

		length = (end-start+1) * width;

		b_inMatrix  = (int*) malloc(sizeof(int) * length);
		g_inMatrix  = (int*) malloc(sizeof(int) * length);
		r_inMatrix  = (int*) malloc(sizeof(int) * length);

		b_outMatrix = (int*) malloc(sizeof(int) * length);
		g_outMatrix = (int*) malloc(sizeof(int) * length);
		r_outMatrix = (int*) malloc(sizeof(int) * length);
		
		MPI_Recv(&b_inMatrix[0], length, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&g_inMatrix[0], length, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&r_inMatrix[0], length, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);

		conv(b_inMatrix, b_outMatrix, (end-start), width);
		conv(g_inMatrix, g_outMatrix, (end-start), width);
		conv(r_inMatrix, r_outMatrix, (end-start), width);

		MPI_Send(&b_outMatrix[0], length, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
		MPI_Send(&g_outMatrix[0], length, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
		MPI_Send(&r_outMatrix[0], length, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	
	return 0;
}


void showImageProperties(IplImage *image)
{
        if(image)
        {
                printf("Width: %d\n", image->width);
                printf("Height: %d\n", image->height);
                printf("Channels: %d\n", image->nChannels);
        } else {
                printf("Image is NULL\n");
        }
}

void printMatrix(int* matrix, int height, int width)
{
	int i,j;

	for( i = 0; i < height; i++ )
        {
		for( j = 0; j < width; j++)
                {
                	printf("%5d ", matrix[i * width + j]);
                }
                printf("\n");
        }
}

int *getMatrix(IplImage *image, int start, int end, int channel) 
{
	int i, j;

	int *matrix = (int*) malloc(sizeof(int) * image->width * (end-start));

	for( i = 0; start < end; i++, start++ ) 
		for( j = 0; j < image->width; j++ ) 
			matrix[i * image->width + j] = cvGet2D(image, start, j).val[channel];

	return matrix;
}
