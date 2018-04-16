#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include<iostream>


using namespace cv;
using namespace std;


void gaussianBlur();
void habijabi();
void contrastStreching();
void gammaCorrection();
void negative();
void grayScale();
void thresholding();
void unsharpFiltering();
void highBoostFiltering();
void errosion();
Mat errosionReturn(Mat input,int maskSize);
void noiseRemoval();
Mat dialation(Mat input,int maskSize);
void boarderExtraction();
Mat showHistogram(Mat input);
void histrogram();
Mat equalizeHistrogram(Mat inputImageMat);
void histogramMatching();
Mat showHistogramF(Mat inputImageMat);
void laplacian();
void discreateFourianTransform();


void takeInput(){
	int key = 1;
	do{
		cout<<endl<<"Choose from following : "<<endl<<endl;
		cout<<"		1.Contrast Streching"<<endl;
		cout<<"		2.Gamma Correction"<<endl;
		cout<<"		3.Negative"<<endl;
		cout<<"		4.Grayscale"<<endl;
		cout<<"		5.Thresholding"<<endl;
		cout<<"		6.Gaussian Blur"<<endl;
		cout<<"		7.Unsharp Mask Filtering"<<endl;
		cout<<"		8.High Boost Filtering"<<endl;
		cout<<"		9.Errosion"<<endl;
		cout<<"		10.Border Exratction"<<endl;
		cout<<"		11.Noise removal"<<endl;
		cout<<"		12.Histogram Equalization"<<endl;
		cout<<"		13.Laplacian"<<endl;
		cout<<"		14.Discreate Fourier transform"<<endl;
		cout<<"		15.Exit"<<endl;
		cin>>key;
		switch (key)
		{
		case 1:
			contrastStreching();
			break;
		case 2:
			gammaCorrection();
			break;
		case 3:
			negative();
			break;
		case 4:
			grayScale();
			break;
		case 5:
			thresholding();
			break;
		case 6:
			gaussianBlur();
			break;
		case 7:
			unsharpFiltering();
			break;
		case 8:
			highBoostFiltering();
			break;
		case 9:
			errosion();
			break;
		case 10:
			boarderExtraction();
			break;
		case 11:
			noiseRemoval();
			break;
		case 12:
			histrogram();
			break;
		case 13:
			laplacian();
			break;
		case 14:
			discreateFourianTransform();
			break;
		default:
			break;
		}
	}while(key!=15);
}

int main(){

	takeInput();

	return 0;
}

void discreateFourianTransform(){
	cout<<"Please Drag & drop your image.."<<endl;
    String imagePath;
    cin>>imagePath;
    Mat I = imread(imagePath,0);

	imshow("Input Image" , I   );
	cvWaitKey(0);

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).

    imshow("spectrum magnitude", magI);
    cvWaitKey();
	cvDestroyAllWindows();
	
}

void laplacian(){
	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath,0),dst,dst3,dst2;
	//cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);

	imshow("Input",inputImageMat);
	cvWaitKey(0);

	float laplacianMatrix[9] = {-1,-1,-1,
								-1,8,-1,
								-1,-1,-1};
	Mat kernel(3,3,CV_32F,laplacianMatrix);


	filter2D(inputImageMat,dst,CV_32F,kernel,Point(-1,-1));

	/*
	imshow("filter2D",dst);
	cvWaitKey(0);
	*/
	//dst = abs(dst);

	imshow("filter2D",dst);
	cvWaitKey(0);

	normalize(dst,dst2,0,255,NORM_MINMAX,CV_8U);

	imshow("Normalize",dst2);
	cvWaitKey(0);

	 


	addWeighted(inputImageMat,0.5,dst2,0.5,0,dst3);
	
	normalize(dst3,dst3,0,255,NORM_MINMAX,CV_8U);

	imshow("Final",dst3);
	cvWaitKey(0);

	cvDestroyAllWindows();
	
}

void gaussianBlur(){
	float sigma;
	cout<<"Sigma = ";
	cin>>sigma;

	int maskSize;
	maskSize = (int) sigma*5.0;
	//cout<<maskSize;
	if(maskSize%2 == 0)maskSize++;

	Mat filter;
	filter.create(maskSize,maskSize,CV_32FC1);

	float val;
	float temp[11][11];
	int s = maskSize/2;
	for(int y = -s,y1 = 0;y <= s ;y++,y1++){
		for(int x = -s,x1 = 0 ; x<= s ; x++,x1++){
			val = (1/(2*3.1416*sigma*sigma))*exp ( (-((y*y)+(x*x)))/(2*sigma*sigma) );
			filter.at<float>(x1,y1) = val;
		}
	}

	//imshow("Filter",filter);
	//cvWaitKey(0);

	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath);
	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);
	imshow("Input",inputImageMat);
	cvWaitKey(0);


	float b,c,d;
	for(int y = s  ;y<inputImageMat.rows - s ;y++){
		for(int x = s  ; x<inputImageMat.cols - s; x++){
			b = 0;
			for(int y1 = -s,y2 = 0; y1<=s ;y1++,y2++){
				for(int x1 = -s ,x2=0; x1<=s ; x1++,x2++){
					b += (float) inputImageMat.at<uchar>(y + y1,x + x1) * filter.at<float>(x2,y2);
				}
			}
			inputImageMat.at<uchar>(y,x) = (uchar) b;
		}
	}

	imshow("Output",inputImageMat);
	cvWaitKey(0);
	cvDestroyWindow("Output");
	cvDestroyWindow("Input");
	

}

void contrastStreching(){
	int r1,r2,s1,s2;
	cout<<"Enter r1,s1,r2,s2 (separated by space) : ";
	cin>>r1>>s1>>r2>>s2;
	


	//Take input
	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath);
	

	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);

	imshow("Input",inputImageMat);
	cvWaitKey(0);

	double s,r;
	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			//for(int c=0;c<3;c++){
			s = inputImageMat.at<uchar>(y,x);
				//s= inputImageMat.at<Vec3b>(y,x)[c];
			if(s>=0 && s<=r1){
				r = (s1/r1)*s;
			}
			else if(s>r1 && s<=r2){
				r = ((s2-s1)/(r2-r1))*(s-r1)+s1;
			}
			else if(s>r2 && s<=255){
				r = ((255-s2)/(255-r2))*(s-r2)+s2;
			}
			//inputImageMat.at<Vec3b>(y,x)[c]=r;
			//}
		    
			inputImageMat.at<uchar>(y,x) = r;
		}
	}

	imshow("Contrast Streching",inputImageMat);
	cvWaitKey(0);
	cvDestroyWindow("Contrast Streching");
	cvDestroyWindow("Input");
}

void gammaCorrection(){
	double gamma;
	cout<<"Gamma = ";
	cin>>gamma;

	//Take input
	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath,0);
	imshow("Input",inputImageMat);
	cvWaitKey(0);

	//cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);

	double s,r;
	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			s = inputImageMat.at<uchar>(y,x);
			r = std::pow(s,gamma);
			r/=(std::pow(255,gamma));
			r*=255;
			inputImageMat.at<uchar>(y,x) = r;
		}
	}

	imshow("Gamma",inputImageMat);
	cvWaitKey(0);
	cvDestroyWindow("Gamma");
	cvDestroyWindow("Input");

}

void negative(){

	//Take input
	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath,0);
	imshow("Input",inputImageMat);
	cvWaitKey(0);

	int s,r;
	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			s = inputImageMat.at<uchar>(y,x);
			r = 255  - s;
			inputImageMat.at<uchar>(y,x) = r;
		}
	}
	imshow("Negative",inputImageMat);
	cvWaitKey(0); 
	cvDestroyWindow("Negative");
	cvDestroyWindow("Input");
}

void grayScale(){
	//Take input
	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath);
	imshow("Input",inputImageMat);
	cvWaitKey(0);


	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);
	imshow("Grayscale",inputImageMat);
	cvWaitKey(0);
	cvDestroyWindow("Grayscale");
	cvDestroyWindow("Input");
}

void thresholding(){
	//Take input
	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath);
	imshow("Input",inputImageMat);
	cvWaitKey(0);

	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);

	int s,r;
	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			s = inputImageMat.at<uchar>(y,x);
			if(s <= 127 && s >=0){
				r = 0;
			}else{
				r = 255;
			}
			inputImageMat.at<uchar>(y,x) = r;
		}
	}

	imshow("Thresholding",inputImageMat);
	cvWaitKey(0);
	cvDestroyWindow("Thresholding");
	cvDestroyWindow("Input");

}

void habijabi(){
	/*
	IplImage* img = cvLoadImage("D:\\ProjectFile\\codenplay_icon.png");
	//cvNamedWindow ("ex" ,CV_WINDOW_NORMAL);
	Mat m = img;
	//bitwise_not(m,m);
	//cvShowImage("Rksazid",img);
	//imshow("Hello",m);
	Mat grayLena,negativeImgae;
	cvtColor(m,grayLena,CV_BGR2XYZ);
	imshow("p",grayLena);
	cvWaitKey(0);
	//cvReleaseImage(&img);
	//cvDestroyWindow("Rksazid");
	negativeImgae = m;

	String st;
	cout<<"Drag & drop your image\n";
	cin>>st;

	*/
	Mat negativeImgae = imread("E:\\Photos\\picci.jpg");
	Mat m;
	cvtColor(negativeImgae,m,CV_RGB2GRAY);
	int s,r;
	for(int y = 0;y<negativeImgae.rows;y++){
		for(int x = 0 ; x<negativeImgae.cols*3 ; x++){
			s = negativeImgae.at<uchar>(y,x);
			r = 255  - s;
			negativeImgae.at<uchar>(y,x) = r;
		}
	}
	imshow("Negative",negativeImgae);
	cvWaitKey(0);

	for(int y = 0;y<m.rows;y++){
		for(int x = 0 ; x<m.cols ; x++){
			s = m.at<uchar>(y,x);
			if(s <= 127 && s >=0){
				r = 0;
			}else{
				r = 255;
			}
			//r = 255  - s;
			m.at<uchar>(y,x) = r;
		}
	}

	imshow("gray",m);
	cvWaitKey(0);
}

void unsharpFiltering(){
	float sigma;
	cout<<"Sigma = ";
	cin>>sigma;

	int maskSize;
	maskSize = (int) sigma*5.0;
	if(maskSize%2 == 0)maskSize++;

	Mat filter,outputImage,maskMat,unsharpMaskingMat;
	filter.create(maskSize,maskSize,CV_32FC1);

	float val,sigmaSq = sigma*sigma;
	float temp[11][11];
	int s = maskSize/2;
	float normalize = (1/(2*3.1416*sigmaSq));
//	int normalize = 1;
	for(int y = -s,y1 = 0;y <= s ;y++,y1++){
		for(int x = -s,x1 = 0 ; x<= s ; x++,x1++){
			val = normalize*exp ( (-((y*y)+(x*x)))/(2*sigmaSq) );
			filter.at<float>(x1,y1) = val;
		}
	}
	
	//imshow("Mask",filter);
	//cvWaitKey(0);

	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath);
	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);
	imshow("Input",inputImageMat);
	cvWaitKey(0);

	outputImage.create(inputImageMat.rows - s*2,inputImageMat.cols - s*2,inputImageMat.type());
	unsharpMaskingMat = outputImage;

	float b,c,d;
	for(int y = s  ;y<inputImageMat.rows - s ;y++){
		for(int x = s  ; x<inputImageMat.cols - s; x++){
			b = 0;
			for(int y1 = -s,y2 = 0; y1<=s ;y1++,y2++){
				for(int x1 = -s ,x2=0; x1<=s ; x1++,x2++){
					b += (float) inputImageMat.at<uchar>(y + y1,x + x1) * filter.at<float>(x2,y2);
				}
			}
			outputImage.at<uchar>(y-s,x-s) = (uchar) b;
		}
	}


	imshow("Output",outputImage);
	cvWaitKey(0);
	//cvDestroyWindow("Output");

	maskMat.create(outputImage.rows,outputImage.cols,outputImage.type());

	uchar mx = 0;
	for(int y = 0 ; y<outputImage.rows ; y++){
		for(int x = 0; x<outputImage.cols ; x++){
			val = abs(inputImageMat.at<uchar>(y+s,x+s) - outputImage.at<uchar>(y,x));
			maskMat.at<uchar>(y,x) = val;
			if(val > mx)mx = val;
		}
	}
	int mm = mx;
	cout<<mm<<endl;

	for(int y = 0 ; y<outputImage.rows ; y++){
		for(int x = 0; x<outputImage.cols ; x++){
			val = (inputImageMat.at<uchar>(y+s,x+s) + maskMat.at<uchar>(y,x));
			val*=255;
			val/=(255+mm);
			//if(val>255)val = 255;
			unsharpMaskingMat.at<uchar>(y,x) = (uchar)val;
		}
	}

	imshow("Mask",maskMat);
	imshow("Unsharp Masking",unsharpMaskingMat);
	cvWaitKey(0);
	cvDestroyAllWindows();

}

void highBoostFiltering(){
	float sigma;
	sigma = 1.3;
	int k;
	cout<<"K = ";
	cin>>k;

	int maskSize;
	maskSize = (int) sigma*5.0;
	if(maskSize%2 == 0)maskSize++;

	Mat filter,outputImage,maskMat,unsharpMaskingMat;
	filter.create(maskSize,maskSize,CV_32FC1);

	float val,sigmaSq = sigma*sigma;
	int s = maskSize/2;
	float normalize = (1/(2*3.1416*sigmaSq));
//	int normalize = 1;
	for(int y = -s,y1 = 0;y <= s ;y++,y1++){
		for(int x = -s,x1 = 0 ; x<= s ; x++,x1++){
			val = normalize*exp ( (-((y*y)+(x*x)))/(2*sigmaSq) );
			filter.at<float>(x1,y1) = val;
		}
	}
	
	

	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath);
	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);
	imshow("Input",inputImageMat);
	cvWaitKey(0);

	outputImage.create(inputImageMat.rows - s*2,inputImageMat.cols - s*2,inputImageMat.type());
	unsharpMaskingMat = outputImage;

	float b,c,d;
	for(int y = s  ;y<inputImageMat.rows - s ;y++){
		for(int x = s  ; x<inputImageMat.cols - s; x++){
			b = 0;
			for(int y1 = -s,y2 = 0; y1<=s ;y1++,y2++){
				for(int x1 = -s ,x2=0; x1<=s ; x1++,x2++){
					b += (float) inputImageMat.at<uchar>(y + y1,x + x1) * filter.at<float>(x2,y2);
				}
			}
			outputImage.at<uchar>(y-s,x-s) = (uchar) b;
		}
	}


	imshow("Output",outputImage);
	cvWaitKey(0);
	//cvDestroyWindow("Output");

	maskMat.create(outputImage.rows,outputImage.cols,outputImage.type());

	uchar mx = 0;
	for(int y = 0 ; y<outputImage.rows ; y++){
		for(int x = 0; x<outputImage.cols ; x++){
			val = k*abs(inputImageMat.at<uchar>(y+s,x+s) - outputImage.at<uchar>(y,x));
			maskMat.at<uchar>(y,x) = val;
			if(val > mx)mx = val;
		}
	}
	int mm = mx;
	cout<<mm<<endl;

	for(int y = 0 ; y<outputImage.rows ; y++){
		for(int x = 0; x<outputImage.cols ; x++){
			val = (inputImageMat.at<uchar>(y+s,x+s) + maskMat.at<uchar>(y,x));
			val*=255;
			val/=(255+mm);
			//if(val>255)val = 255;
			unsharpMaskingMat.at<uchar>(y,x) = (uchar)val;
		}
	}

	imshow("Mask",maskMat);
	imshow("Unsharp Masking",unsharpMaskingMat);
	cvWaitKey(0);
	cvDestroyAllWindows();
}

void errosion(){
	int maskSize;
	cout<<"Mask size  = ";
	cin>>maskSize;

	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath),outputMat;
	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);

	int r,s;
	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			s = inputImageMat.at<uchar>(y,x);
			if(s <= 127 && s >=0){
				r = 0;
			}else{
				r = 255;
			}
			inputImageMat.at<uchar>(y,x) = r;
		}
	}

	outputMat = inputImageMat;
	imshow("Input",inputImageMat);
	cvWaitKey(0);
	
	copyMakeBorder(inputImageMat,outputMat,maskSize/2,maskSize/2,maskSize/2,maskSize/2,0,0);
	copyMakeBorder(inputImageMat,inputImageMat,maskSize/2,maskSize/2,maskSize/2,maskSize/2,0,0);
	
	bool flag = true;
	s = maskSize/2;
	for(int y = s  ;y<outputMat.rows - s ;y++){
		for(int x = s  ; x<outputMat.cols - s; x++){
			flag = true;
			for(int y1 = -s,y2 = 0; y1<=s ;y1++,y2++){
				for(int x1 = -s ,x2=0; x1<=s ; x1++,x2++){
					if(inputImageMat.at<uchar>(y + y1 ,x + x1) != 255){
						flag = false;
					}
				}
			}
			if(flag == false){
				outputMat.at<uchar>(y,x) = 0;
			}
		}
	}

	imshow("Output",outputMat);
	cvWaitKey(0);
	cvDestroyAllWindows();


}

void boarderExtraction(){
	int maskSize;
	cout<<"Mask size  = ";
	cin>>maskSize;

	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath),outputMat;
	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);

	int r,s;
	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			s = inputImageMat.at<uchar>(y,x);
			if(s <= 127 && s >=0){
				r = 0;
			}else{
				r = 255;
			}
			inputImageMat.at<uchar>(y,x) = r;
		}
	}

	outputMat = inputImageMat;
	imshow("Input",inputImageMat);
	cvWaitKey(0);
	
	copyMakeBorder(inputImageMat,outputMat,maskSize/2,maskSize/2,maskSize/2,maskSize/2,0,0);
	copyMakeBorder(inputImageMat,inputImageMat,maskSize/2,maskSize/2,maskSize/2,maskSize/2,0,0);
	
	bool flag = true;
	s = maskSize/2;
	for(int y = s  ;y<outputMat.rows - s ;y++){
		for(int x = s  ; x<outputMat.cols - s; x++){
			flag = true;
			for(int y1 = -s,y2 = 0; y1<=s ;y1++,y2++){
				for(int x1 = -s ,x2=0; x1<=s ; x1++,x2++){
					if(inputImageMat.at<uchar>(y + y1 ,x + x1) != 255){
						flag = false;
					}
				}
			}
			if(flag == false){
				outputMat.at<uchar>(y,x) = 0;
			}
		}
	}

	for(int y = 0  ;y<outputMat.rows ;y++){
		for(int x = s  ; x<outputMat.cols ; x++){
			outputMat.at<uchar>(y,x) = abs(outputMat.at<uchar>(y,x)-inputImageMat.at<uchar>(y,x));
		}
	}

	imshow("Output",outputMat);
	cvWaitKey(0);
	cvDestroyAllWindows();

}

Mat errosionReturn(Mat inputImageMat,int maskSize){
	Mat outputMat;

	outputMat = inputImageMat;
	//imshow("Input",inputImageMat);
	//cvWaitKey(0);
	
	copyMakeBorder(inputImageMat,outputMat,maskSize/2,maskSize/2,maskSize/2,maskSize/2,0,0);
	copyMakeBorder(inputImageMat,inputImageMat,maskSize/2,maskSize/2,maskSize/2,maskSize/2,0,0);
	
	bool flag = true;
	int s = maskSize/2;
	for(int y = s  ;y<outputMat.rows - s ;y++){
		for(int x = s  ; x<outputMat.cols - s; x++){
			flag = true;
			for(int y1 = -s,y2 = 0; y1<=s ;y1++,y2++){
				for(int x1 = -s ,x2=0; x1<=s ; x1++,x2++){
					if(inputImageMat.at<uchar>(y + y1 ,x + x1) != 255){
						flag = false;
					}
				}
			}
			if(flag == false){
				outputMat.at<uchar>(y,x) = 0;
			}
		}
	}

	return outputMat;
}

Mat dialation(Mat inputImageMat,int maskSize){
	Mat outputMat;


	outputMat = inputImageMat;
	//imshow("Input",inputImageMat);
	//cvWaitKey(0);
	
	copyMakeBorder(inputImageMat,outputMat,maskSize/2,maskSize/2,maskSize/2,maskSize/2,0,0);
	copyMakeBorder(inputImageMat,inputImageMat,maskSize/2,maskSize/2,maskSize/2,maskSize/2,0,0);
	
	bool flag = true;
	int s = maskSize/2;
	for(int y = s  ;y<outputMat.rows - s ;y++){
		for(int x = s  ; x<outputMat.cols - s; x++){
			flag = true;
			for(int y1 = -s,y2 = 0; y1<=s ;y1++,y2++){
				for(int x1 = -s ,x2=0; x1<=s ; x1++,x2++){
					if(inputImageMat.at<uchar>(y + y1 ,x + x1) == 255){
						flag = false;
					}
				}
			}
			if(flag == false){
				outputMat.at<uchar>(y,x) = 255;
			}
		}
	}

	return outputMat;
}

void noiseRemoval(){
	int maskSize;
	cout<<"Mask size  = ";
	cin>>maskSize;

	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath),outputMat;
	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);

	int r,s;
	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			s = inputImageMat.at<uchar>(y,x);
			if(s <= 127 && s >=0){
				r = 0;
			}else{
				r = 255;
			}
			inputImageMat.at<uchar>(y,x) = r;
		}
	}

	imshow("Thresholded Input",inputImageMat);
	cvWaitKey(0);


	outputMat = errosionReturn(inputImageMat,maskSize);
	imshow("Errosion 1",outputMat);
	cvWaitKey(0);

	outputMat = dialation(outputMat,maskSize);
	imshow("Dialation 1",outputMat);
	cvWaitKey(0);

	outputMat = dialation(outputMat,maskSize);
	imshow("Dilalation 2",outputMat);
	cvWaitKey(0);

	outputMat = errosionReturn(outputMat,maskSize);
	imshow("Errosion 2",outputMat);
	cvWaitKey(0);

	cvDestroyAllWindows();
}

Mat showHistogram(Mat inputImageMat){

	int frequency[256] =  {0};
	float normalizeFrequency[256]={0};
	int s,mx=0;
	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			s = (int) inputImageMat.at<uchar>(y,x);
			frequency[s]++;

			if(frequency[s]>mx){
				mx = frequency[s];
			}
		}
	}

	//cout<<mx<<endl;
	
	for(int i = 0 ; i<256 ;i++){
		normalizeFrequency[i] = ceil(((float)frequency[i]/mx)*300);
	}
	
	/*
	for(int i = 0;i<256;i++){
		cout<<frequency[i]<<" ";
	}
	*/

	Mat histrogramMat;
	histrogramMat.create(300,256,CV_8UC1);
	histrogramMat.setTo(0);

	for(int y = histrogramMat.rows-1 ; y>0;y--){
		for(int x = normalizeFrequency[y] -1 ; x>0 ; x--){
			histrogramMat.at<uchar>(300-x,y) = 255;
		}
	}
	
	return histrogramMat;

}

Mat equalizeHistrogram(Mat inputImageMat){
	int frequency[256] =  {0};
	float normalizeFrequency[256]={0};
	memset(normalizeFrequency,0.0,sizeof normalizeFrequency);
	memset(frequency,0,sizeof frequency);
	int s,mx=0;
	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			s = (int) inputImageMat.at<uchar>(y,x);
			frequency[s]++;
		}
	}

	normalizeFrequency[0] =  frequency[0];

	for(int i = 1;i<256;i++){
		normalizeFrequency[i] = normalizeFrequency[i-1] + frequency[i];
	}

	for(int i = 1;i<256;i++){
		normalizeFrequency[i] /= (float)(inputImageMat.rows*inputImageMat.cols);
	}
	

	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			s = (int) inputImageMat.at<uchar>(y,x);
			inputImageMat.at<uchar>(y,x) =  (uchar) floor(normalizeFrequency[s]*255);
		}
	}

	return inputImageMat;
}

void histrogram(){
	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath),outputMat;
	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);

	Mat histrogramMat;
	histrogramMat.create(300,256,CV_8UC1);
	histrogramMat.setTo(0);

	imshow("Input",inputImageMat);
	cvWaitKey(0);

	histrogramMat = showHistogram(inputImageMat);
	imshow("input Histrogram",histrogramMat);
	cvWaitKey(0);


	inputImageMat = equalizeHistrogram(inputImageMat);

	imshow("Equalize",inputImageMat);
	cvWaitKey(0);

	histrogramMat = showHistogram(inputImageMat);
	imshow("Equalize Histrogram",histrogramMat);
	cvWaitKey(0);

	cvDestroyAllWindows();
}

Mat showHistogramF(Mat inputImageMat){

	int frequency[256] =  {0};
	float normalizeFrequency[256]={0};
	int s,mx=0;
	for(int y = 0;y<inputImageMat.rows;y++){
		for(int x = 0 ; x<inputImageMat.cols ; x++){
			s = (int) inputImageMat.at<float>(y,x);
			frequency[s]++;

			if(frequency[s]>mx){
				mx = frequency[s];
			}
		}
	}

	cout<<mx<<endl;
	
	for(int i = 0 ; i<256 ;i++){
		normalizeFrequency[i] = ceil(((float)frequency[i]/mx)*300);
	}
	

	for(int i = 0;i<256;i++){
		cout<<frequency[i]<<" ";
	}

	Mat histrogramMat;
	histrogramMat.create(300,256,CV_8UC1);
	histrogramMat.setTo(0);

	for(int y = histrogramMat.rows-1 ; y>0;y--){
		for(int x = normalizeFrequency[y] -1 ; x>0 ; x--){
			histrogramMat.at<uchar>(300-x,y) = 255;
		}
	}
	
	return histrogramMat;

}

void histogramMatching(){
	cout<<"Please Drag & drop your image.."<<endl;
	String imagePath;
	cin>>imagePath;
	Mat inputImageMat = imread(imagePath),outputMat;
	cvtColor(inputImageMat,inputImageMat,CV_RGB2GRAY);

	float sigma;
	cout<<"Sigma = ";
	cin>>sigma;

	int maskSize;
	maskSize = (int) sigma*5;
	//cout<<maskSize;
	if(maskSize%2 == 0)maskSize++;

	Mat filter;
	filter.create(maskSize,maskSize,CV_32FC1);

	float val;
	float temp[11][11];
	float mxx = 1000000; 
	int s = maskSize/2;
	for(int y = -s,y1 = 0;y <= s ;y++,y1++){
		for(int x = -s,x1 = 0 ; x<= s ; x++,x1++){
			val = exp ( (-((y*y)+(x*x)))/(2*sigma*sigma) );
			mxx = min(mxx,val);
			filter.at<float>(x1,y1) = val;
		}
	}

	Mat Gauss;
	Gauss.create(filter.rows,filter.cols,inputImageMat.type());
	Gauss.setTo(0);

	for(int y = -s,y1 = 0;y <= s ;y++,y1++){
		for(int x = -s,x1 = 0 ; x<= s ; x++,x1++){
			//Gauss.at<uchar>(x1,y1) = (uchar)(filter.at<float>(x1,y1)/mxx);
			cout<<filter.at<float>(x1,y1)/mxx<<" ";
		}
		cout<<endl;
	}
	

	imshow("Gaussian",Gauss);
	cvWaitKey(0);

	Mat histoGauss = showHistogramF(Gauss);

	imshow("HistoGauss",histoGauss);
	cvWaitKey(0);
	
}

/*
float* p;
float* q;
for( int i = 0; i < image.rows; ++i)
{
    p = image.ptr< float >( i );
    q = image2.ptr< float >( i ); // I guess image2 is the same size as image1
    // and both image are in float!
    for ( int j = 0; j < image.cols; ++j)
    {
        // Do whatever you want
        p[ j ] = 255.f * q[ j ] / 3.5f;
    }
}
*/