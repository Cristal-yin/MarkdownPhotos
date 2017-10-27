#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

//英特尔处理器和其他低端机用户必须翻转头字节。
int ReverseInt(int i) 
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
//读取训练与测试数据
void read_Mnist(string filename, vector<cv::Mat> &vec)
{
	ifstream file (filename, ios::binary);
	if (file.is_open()) 
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		//从文件中读取sizeof(magic_number) 个字符到 &magic_number
		file.read((char*) &magic_number, sizeof(magic_number)); 
		magic_number = ReverseInt(magic_number);
		//获取训练或测试image的个数number_of_images 
		file.read((char*) &number_of_images,sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);  
		//获取训练或测试图像的高度Heigh
		file.read((char*) &n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);                
		//获取训练或测试图像的宽度Width
		file.read((char*) &n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		//获取第i幅图像，保存到vec中
		for(int i = 0; i < number_of_images; ++i)
		{
			cv::Mat tp = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);
			for(int r = 0; r < n_rows; ++r) 
			{
				for(int c = 0; c < n_cols; ++c) 
				{
					unsigned char temp = 0;
					file.read((char*) &temp, sizeof(temp));
					tp.at<uchar>(r, c) = (int) temp;
				}
			}
			vec.push_back(tp);  
		}
	}
}

//读取训练与测试标签
void read_Mnist_Label(string filename, vector<int> &vec)
{
	ifstream file (filename, ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*) &magic_number, sizeof(magic_number));
	   	magic_number = ReverseInt(magic_number);

		file.read((char*) &number_of_images,sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		for(int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*) &temp, sizeof(temp));
			vec[i]= (int)temp;
		}
	}
}

string GetImageName(int number, int arr[])
{
	string str1, str2;

	for (int i = 0; i < 10; i++) {
		if (number == i) {
			arr[i]++;
			char ch1[10];  
			sprintf(ch1, "%d", arr[i]);   
			str1 = std::string(ch1); 

			if (arr[i] < 10) {
				str1 = "0000" + str1;
			} else if (arr[i] < 100) {
				str1 = "000" + str1;
			} else if (arr[i] < 1000) {
				str1 = "00" + str1;
			} else if (arr[i] < 10000) {
				str1 = "0" + str1;
			}

			break;
		}
	}

	char ch2[10];
	sprintf(ch2, "%d", number);
	str2 = std::string(ch2);

	str2 = str2 + "_" + str1;

	return str2;
}

int main()
{
//测试数据和测试标签
	//读取测试数据 转换为Mat
	string filename_test_images = "~/caffe/data/mnist/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
	int number_of_test_images = 10000;   //测试数据10000个
    vector<cv::Mat> vec_test_images;

    read_Mnist(filename_test_images, vec_test_images);

	//读取测试标签 转换为vector
    string filename_test_labels = "~/caffe/data/mnist/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte";
    vector<int> vec_test_labels(number_of_test_images);

    read_Mnist_Label(filename_test_labels, vec_test_labels);

	if (vec_test_images.size() != vec_test_labels.size()) {
		cout<<"parse MNIST test file error"<<endl;
		return -1;
	}

	//保存测试图像
	int count_digits[10];
	for (int i = 0; i < 10; i++)
		count_digits[i] = 0;

	string save_test_images_path = "~/caffe/data/mnist/test_images/";   //保存路径

	for (int i = 0; i < vec_test_images.size(); i++) 
	{
		int number = vec_test_labels[i];
		string image_name = GetImageName(number, count_digits);
		image_name = save_test_images_path + image_name + ".jpg";

		cv::imwrite(image_name, vec_test_images[i]);
	}

//训练数据与训练标签
	//read MNIST image into OpenCV Mat vector
	string filename_train_images = "~/caffe/data/mnist/train-images-idx3-ubyte/train-images.idx3-ubyte";
	int number_of_train_images = 60000;
	vector<cv::Mat> vec_train_images;

	read_Mnist(filename_train_images, vec_train_images);

	//read MNIST label into int vector
	string filename_train_labels = "~/caffe/data/mnist/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
	vector<int> vec_train_labels(number_of_train_images);

	read_Mnist_Label(filename_train_labels, vec_train_labels);

	if (vec_train_images.size() != vec_train_labels.size()) {
		cout<<"parse MNIST train file error"<<endl;
		return -1;
	}

	//save train images
	for (int i = 0; i < 10; i++)
		count_digits[i] = 0;

	string save_train_images_path = "~/caffe/data/mnist/train_images/"; //保存路径

	for (int i = 0; i < vec_train_images.size(); i++) {
		int number = vec_train_labels[i];
		string image_name = GetImageName(number, count_digits);
		image_name = save_train_images_path + image_name + ".jpg";

		cv::imwrite(image_name, vec_train_images[i]);
	}

	return 1;
}
