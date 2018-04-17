#include<opencv\cv.h>
#include <opencv\highgui.h>
#include <iostream>		

using namespace cv; 
using namespace std;



void shiftDFT(Mat& fImage )
{
  	Mat tmp, q0, q1, q2, q3;

	fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));

	int cx = fImage.cols/2;
	int cy = fImage.rows/2;


	q0 = fImage(Rect(0, 0, cx, cy));
	q1 = fImage(Rect(cx, 0, cx, cy));
	q2 = fImage(Rect(0, cy, cx, cy));
	q3 = fImage(Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

Mat create_spectrum_magnitude_display(Mat& complexImg, bool rearrange)
{
    Mat planes[2];

    split(complexImg, planes);
    magnitude(planes[0], planes[1], planes[0]);

    Mat mag = (planes[0]).clone();
    mag += Scalar::all(1);
    log(mag, mag);

    if (rearrange)
    {
        // re-arrange the quaderants
        shiftDFT(mag);
    }

    normalize(mag, mag, 0, 1, CV_MINMAX);

    return mag;

}

void create_ideal_lowpass_filter(Mat &dft_Filter, int D)
{
	Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

	Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
	double radius;

	for(int i = 0; i < dft_Filter.rows; i++)
	{
		for(int j = 0; j < dft_Filter.cols; j++)
		{
			radius = (double) sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));
			if(radius>D){
				tmp.at<float>(i,j) = (float)0;
			}else{
				tmp.at<float>(i,j) = (float)1;
			}
						
		}
	}

    Mat toMerge[] = {tmp, tmp};
	merge(toMerge, 2, dft_Filter);
}

void create_butterworth_lowpass_filter(Mat &dft_Filter, int D, int n)
{
	Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

	Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
	double radius;


	for(int i = 0; i < dft_Filter.rows; i++)
	{
		for(int j = 0; j < dft_Filter.cols; j++)
		{
			radius = (double) sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));
			tmp.at<float>(i,j) = (float)
						( 1 / (1 + pow((double) (radius /  D), (double) (2 * n))));
		}
	}

    Mat toMerge[] = {tmp, tmp};
	merge(toMerge, 2, dft_Filter);
}

void create_ideal_highpass_filter(Mat &dft_Filter, int D)
{
	Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

	Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
	double radius;

	for(int i = 0; i < dft_Filter.rows; i++)
	{
		for(int j = 0; j < dft_Filter.cols; j++)
		{
			radius = (double) sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));
			if(radius>D){
				tmp.at<float>(i,j) = (float)1;
			}else{
				tmp.at<float>(i,j) = (float)0;
			}
						
		}
	}

    Mat toMerge[] = {tmp, tmp};
	merge(toMerge, 2, dft_Filter);
}


int key,radius=30,n=2;
void choice(){
	cout<<"Please choose from the following : "<<endl;
	cout<<"		1.Ideal Low pass filter"<<endl;
	cout<<"		2.Butterworth Low pass filter"<<endl;
	cout<<"		3.Ideal High pass filter"<<endl;

	cin>>key;

	cout<<"Radius : ";
	cin>>radius;

	if(key==2){
		cout<<"Order : ";
		cin>>n;
	}
}

void solve(){
	Mat imgOutput;	
	Mat padded;		
	Mat complexImg, filter, filterOutput;
	Mat planes[2], mag;
	
	choice();

	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat img = imread(imagePath,0);
                                
	int N, M ;
			

     M = getOptimalDFTSize( img.rows );
     N = getOptimalDFTSize( img.cols );

	imshow("Input",img);
	cvWaitKey(0);

	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	planes[0] = Mat_<float>(padded);
	planes[1] = Mat::zeros(padded.size(), CV_32F);
	merge(planes, 2, complexImg);
	
	dft(complexImg, complexImg);
	    
	filter = complexImg.clone();
	switch (key)
	{
	case 1:
		create_ideal_lowpass_filter(filter,radius);
		break;
	case 2:
		create_butterworth_lowpass_filter(filter,radius,n);
		break;
	case 3:
		create_ideal_highpass_filter(filter, radius);
		break;
	default:
		break;
	}
	
				    
	shiftDFT(complexImg);
	mulSpectrums(complexImg, filter, complexImg, 0);
			
	shiftDFT(complexImg);

    mag = create_spectrum_magnitude_display(complexImg, true);

    idft(complexImg, complexImg);

    split(complexImg, planes);
    normalize(planes[0], imgOutput, 0, 1, CV_MINMAX);

           
    split(filter, planes);
    normalize(planes[0], filterOutput, 0, 1, CV_MINMAX);


    imshow("Magnitude Image (log transformed)", mag);
    imshow("Ideal high pass filter", imgOutput);
    imshow("Filter", filterOutput);
    cvWaitKey(0);
	cvDestroyAllWindows();
}

int main()
{
	while(true){
		solve();
	}
    return 0;
}
