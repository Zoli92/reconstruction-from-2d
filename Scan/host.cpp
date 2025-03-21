//Use cl::vector instead of STL version
#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include "stb_image.h"
#include "stb_image_write.h"
using namespace cl;
int IMAGE_WIDTH = 1390;
int IMAGE_HEIGHT = 1110;
int FOCAL_LENGTH = 3740;
int BASELINE = 160; //mm

struct rgba
{
	unsigned char r, g, b, a;
	rgba() : r(0), g(0), b(0), a(0) {}
	rgba(unsigned char r, unsigned char g, unsigned char b, unsigned char a) : r(r), g(g), b(b), a(a) {}
};
struct RGBAImage
{
	int width = IMAGE_WIDTH;
	int height = IMAGE_HEIGHT;
	rgba* elements;
	size_t stride_in_bytes;

};
unsigned round_up_div(unsigned a, unsigned b) {
	return static_cast<int>(ceil((double)a / b));
}
bool loadImage(const char* filename, RGBAImage& img);
void saveImage(const char* filename, const RGBAImage& img);
std::vector<float> convertToGrayscale(const RGBAImage& img);
void saveDisparityMap(const char* filename, const std::vector<float>& disp_vector, int width, int height);
void convertToPLY(std::vector<float> disparity_map, const char* filename);
#include <oclutils.hpp>

int main() {

	try {
#pragma region Initialize GPU

		Context context;
		if (!oclCreateContextBy(context, "nvidia")) {
			throw Error(CL_INVALID_CONTEXT, "Failed to create a valid context!");
		}

		// Query devices from the context
		vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

		// Create a command queue and use the first device
		CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

		// Read source file
		auto sourceCode = 	oclReadSourcesFromFile("kernels.cl");
		Program::Sources sources(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

		// Make program of the source code in the context
		Program program(context, sources);

		// Build program for these specific devices
		try {
			program.build(devices);
		}
		catch (Error error) {
			oclPrintError(error);
			std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
			std::cerr << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
			std::cerr << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
			throw error;
		}

		// Make kernels
		Kernel kernel_template(program, "templateMatching");
		Kernel kernel_calculateDissim(program, "calculateDissim");
		Kernel kernel_calculateC(program, "calculateC");
		Kernel kernel_MSE1(program, "Reduce"), kernel_MSE2(program, "Reduce");
		Kernel kernel_pointMSE(program, "pointMSE");
		Kernel kernel_pointCloud(program, "pointCloud");
#pragma endregion
		// Load both images
		RGBAImage left_image;
		loadImage("images/view1.png", left_image);
		RGBAImage right_image;
		loadImage("images/view5.png", right_image);

		// Convert to grayscale	
		std::vector<float> left_grayscale = convertToGrayscale(left_image);
		std::vector<float> right_grayscale = convertToGrayscale(right_image);



#pragma region Execute kernel

			// Create memory buffers
			Buffer buffer_1(context, CL_MEM_READ_WRITE, left_image.width*left_image.height*sizeof(float));
			Buffer buffer_2(context, CL_MEM_READ_WRITE, left_image.width * left_image.height * sizeof(float));
			Buffer disparity_buffer(context, CL_MEM_WRITE_ONLY, left_image.width * left_image.height * sizeof(float));
			// Copy input to the memory buffer
			queue.enqueueWriteBuffer(buffer_1, CL_TRUE, 0, left_image.width * left_image.height * sizeof(float), left_grayscale.data());
			queue.enqueueWriteBuffer(buffer_2, CL_TRUE, 0, left_image.width * left_image.height * sizeof(float), right_grayscale.data());

			double time = 0;
			//Set template matching kernel args
			Event operation;
			kernel_template.setArg(0, buffer_1);
			kernel_template.setArg(1, buffer_2);
			kernel_template.setArg(2, disparity_buffer);
			kernel_template.setArg(3, left_image.width);
			kernel_template.setArg(4, left_image.height);
			kernel_template.setArg(5, 4);
			
			//Create result vector
			std::vector<float> result_vector(IMAGE_WIDTH*IMAGE_HEIGHT);
			queue.enqueueNDRangeKernel(kernel_template, cl::NullRange, cl::NDRange(left_image.width, left_image.height), cl::NullRange, nullptr, &operation);
			
			//Read the result
			queue.enqueueReadBuffer(disparity_buffer, CL_TRUE, 0, left_image.width * left_image.height * sizeof(float), &result_vector[0]);
			time += oclGetTiming(operation);
			std::cout << time << std::endl;
			//Save as image
			saveDisparityMap("images/output.png", result_vector, left_image.width, left_image.height);
			std::cout << "Image saved as output.png" << std::endl;
			

			//Load original label for comparison
			RGBAImage original_image;
			loadImage("images/label_4.png", original_image);
			std::vector<float> label_vector = convertToGrayscale(original_image);
			Buffer label_buffer(context, CL_MEM_READ_WRITE, left_image.width * left_image.height * sizeof(float));
			queue.enqueueWriteBuffer(label_buffer, CL_TRUE, 0, left_image.width * left_image.height * sizeof(float), &label_vector[0], nullptr, nullptr);

			//Create buffers for calculating the squared differences
			Buffer result_buffer(context, CL_MEM_READ_WRITE, left_image.width * left_image.height * sizeof(float));
			Buffer individualMSE(context, CL_MEM_READ_WRITE, left_image.width * left_image.height * sizeof(float));
			
			kernel_pointMSE.setArg(0, label_buffer);
			kernel_pointMSE.setArg(1, disparity_buffer);
			kernel_pointMSE.setArg(2, individualMSE);
			kernel_pointMSE.setArg(3, left_image.width);
			kernel_pointMSE.setArg(4, left_image.height);

			//Calculate the squared differences for every pixel
			queue.enqueueNDRangeKernel(kernel_pointMSE, cl::NullRange, cl::NDRange(left_image.width, left_image.height), cl::NullRange, nullptr, nullptr);

			const unsigned int GROUP_SIZE = 128;

			//Calculate the mean of squared differences
			kernel_MSE1.setArg(0, individualMSE);
			kernel_MSE1.setArg(1, result_buffer);
			kernel_MSE1.setArg(2, GROUP_SIZE * sizeof(float), nullptr);

			kernel_MSE2.setArg(0, result_buffer);
			kernel_MSE2.setArg(1, individualMSE);
			kernel_MSE2.setArg(2, GROUP_SIZE * sizeof(float), nullptr);


			cl_ulong round = 0;
			for (int size = left_image.width * left_image.height; size > 1; size = round_up_div(size, GROUP_SIZE), ++round)
			{
				Event operation2;
				int t1 = round_up_div(size, GROUP_SIZE) * GROUP_SIZE;
				queue.enqueueNDRangeKernel(round % 2 == 0 ? kernel_MSE1 : kernel_MSE2, cl::NullRange, t1, GROUP_SIZE, nullptr, &operation2);
			}
			float MSE;
			queue.enqueueReadBuffer(round % 2 == 0 ? individualMSE : result_buffer, CL_TRUE, 0, sizeof(float), &MSE);
			
			//Calculate peak signal-to-noise ratio
			float PSNR = 10 * log10(255 * 255 / MSE);
			std::cout << "MEAN SQUARED ERROR: " << MSE << std::endl << "PEAK SIGNAL TO NOISE RATIO: " << PSNR << std::endl;


			//Converting into 3d pointcloud

			std::vector<float> pointcloud(IMAGE_WIDTH* IMAGE_HEIGHT * 4);
			Buffer pointcloud_buffer(context, CL_MEM_READ_WRITE, IMAGE_WIDTH * IMAGE_HEIGHT * 4 * sizeof(float));
			kernel_pointCloud.setArg(0, label_buffer);
			kernel_pointCloud.setArg(1, pointcloud_buffer);
			kernel_pointCloud.setArg(2, FOCAL_LENGTH);
			kernel_pointCloud.setArg(3, BASELINE);
			kernel_pointCloud.setArg(4, IMAGE_WIDTH);

			queue.enqueueNDRangeKernel(kernel_pointCloud, cl::NullRange, cl::NDRange(IMAGE_WIDTH, IMAGE_HEIGHT), cl::NullRange, nullptr, nullptr);

			queue.enqueueReadBuffer(pointcloud_buffer, CL_TRUE, 0, IMAGE_WIDTH * IMAGE_HEIGHT * 4 * sizeof(float), &pointcloud[0], nullptr, nullptr);
			
			//Write the .ply file
			std::ofstream plyFile("images/output.ply");
			plyFile << "ply\nformat ascii 1.0\n";
			plyFile << "element vertex " << IMAGE_WIDTH * IMAGE_HEIGHT << "\n";
			plyFile << "property float x\nproperty float y\nproperty float z\n";
			plyFile << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";

			for (int i = 0; i < IMAGE_HEIGHT; i++) {
				for (int j = 0; j < IMAGE_WIDTH; j++) {
					int idx = i * IMAGE_WIDTH + j;
					float X = pointcloud[idx * 4 + 0];
					float Y = pointcloud[idx * 4 + 1];
					float Z = pointcloud[idx * 4 + 2];
					float d = pointcloud[idx * 4 + 3];

					
					if (d >= 200 || d <= 0)
					{
						plyFile << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << "\n";

					}else
					if (d < 100)
					{
						plyFile << X << " " << Y << " " << Z << " " <<  0 << " " << 2*(d+27) << " " << 255 << "\n";
					}
					else if (d < 150)
					{
						plyFile << X << " " << Y << " " << Z << " " << 2*(d - 23) << " " << 255 << " " << 255 - 2*(d-23) << "\n";
					}
					else {
						plyFile << X << " " << Y << " " << Z << " " << 255 << " " << 255-(d-150) *2<< " " << 0 << "\n";
					}
					
				}
			}
			plyFile.close();

			std::cout << "Point cloud saved to output.ply" << std::endl;

			return 0;


			 //0-match, 1-left,2-right
			//Buffer dissim_matrix(context, CL_MEM_READ_WRITE, IMAGE_WIDTH * IMAGE_WIDTH * sizeof(float));
			//Buffer cost_matrix(context, CL_MEM_READ_WRITE, IMAGE_WIDTH * IMAGE_WIDTH * sizeof(float));
			//Buffer node_matrix(context, CL_MEM_READ_WRITE, IMAGE_WIDTH * IMAGE_WIDTH * sizeof(int));
			//std::vector<float> zeros(IMAGE_WIDTH * IMAGE_WIDTH);
			//Buffer disparity_matrix(context, CL_MEM_READ_WRITE, 1390 * 1110 * sizeof(float));
			//int step = 0;
			//std::vector<float> disparities(IMAGE_WIDTH * IMAGE_HEIGHT);
			//for(int k = 0; k < IMAGE_HEIGHT; k++)
			//{
			//	std::cout << k << "\n";
			//	queue.enqueueWriteBuffer(dissim_matrix, CL_TRUE, 0, IMAGE_WIDTH*IMAGE_WIDTH*sizeof(float),&zeros[0], nullptr, nullptr);
			//	kernel_calculateDissim.setArg(0, buffer_1);
			//	kernel_calculateDissim.setArg(1, buffer_2);
			//	kernel_calculateDissim.setArg(2, dissim_matrix);
			//	kernel_calculateDissim.setArg(3, IMAGE_WIDTH);
			//	kernel_calculateDissim.setArg(4, IMAGE_HEIGHT);
			//	kernel_calculateDissim.setArg(5, 5);
			//	kernel_calculateDissim.setArg(6, k);

			//	queue.enqueueNDRangeKernel(kernel_calculateDissim, cl::NullRange, cl::NDRange(IMAGE_WIDTH,IMAGE_WIDTH), cl::NullRange, nullptr, &operation);
			//	for (int i = 0; i <= 2*IMAGE_WIDTH-1; i++)
			//	{
			//		
			//		
			//		kernel_calculateC.setArg(0, buffer_1);
			//		kernel_calculateC.setArg(1, buffer_2);
			//		kernel_calculateC.setArg(2, dissim_matrix);
			//		kernel_calculateC.setArg(3, i);
			//		kernel_calculateC.setArg(4, cost_matrix);
			//		kernel_calculateC.setArg(5, node_matrix);
			//		kernel_calculateC.setArg(7, 0);

			//		if(i <= IMAGE_WIDTH)
			//		{
			//			step = i;
			//		}
			//		else
			//		{
			//			step--;
			//			kernel_calculateC.setArg(7, 1);
			//		}
			//		kernel_calculateC.setArg(6, k);
			//		
			//		queue.enqueueNDRangeKernel(kernel_calculateC, cl::NullRange, step + 1, cl::NullRange, nullptr, &operation);
			//	}

			//	std::vector<float> costs(IMAGE_WIDTH * IMAGE_WIDTH);

			//	std::vector<int> nodes(IMAGE_WIDTH* IMAGE_WIDTH);
			//	
			//	queue.enqueueReadBuffer(cost_matrix, CL_TRUE, 0, IMAGE_WIDTH*IMAGE_WIDTH * sizeof(float), &costs[0]);
			//	queue.enqueueReadBuffer(node_matrix, CL_TRUE, 0, IMAGE_WIDTH*IMAGE_WIDTH* sizeof(int), &nodes[0]);

			//	//Megvannak a mátrixok, visszaterjesztés
			//	int x = IMAGE_WIDTH-1;
			//	int y =IMAGE_WIDTH-1;
			//	bool update = true;

			//	while (x > 0 && y > 0)
			//	{
			//		if(update)
			//		{
			//			if (nodes[y * IMAGE_WIDTH + x] == 0)
			//			{
			//				disparities[k * IMAGE_WIDTH + x] = abs(x-y);
			//				
			//			}
			//			else if (nodes[y * IMAGE_WIDTH + x] == 1)
			//			{
			//				disparities[k * IMAGE_WIDTH + x] = 0;
			//			}						
			//		
			//		}
			//		
			//		if (nodes[y * IMAGE_WIDTH + x] == 0)
			//		{
			//			
			//			update = true;
			//			x--;
			//			y--;
			//		}
			//		else if(nodes[y*IMAGE_WIDTH + x] == 1)
			//		{
			//			
			//			update = true;
			//			x--;
			//			
			//		}else if(nodes[y*IMAGE_WIDTH + x] == 2)
			//		{	
			//			update = false;
			//			y--;
			//		}

			//	}
			//	while (x > 0) {
			//		disparities[k * IMAGE_WIDTH + x] = -1;
			//		x--;
			//	}



			//	//std::cout << "\n";
			//}
			//std::cout << "done";
			//saveDisparityMap("pls.png", disparities, left_image.width, left_image.height);
			//

#pragma endregion
			
		}

	catch (Error error) {
		oclPrintError(error);
	}

	std::cin.get();

	return 0;
}

bool loadImage(const char* filename, RGBAImage& img)
{
	int comp;
	stbi_uc* img_uc = stbi_load(filename, &img.width, &img.height, &comp, 4);
	img.elements = new rgba[img.width * img.height];
	img.stride_in_bytes = img.width * sizeof(rgba);
	memcpy(img.elements, img_uc, img.width * img.height * sizeof(rgba));

	return img.elements != nullptr;
}
void saveImage(const char* filename, const RGBAImage& img)
{
	stbi_write_png(filename, img.width, img.height, 4, img.elements, img.stride_in_bytes);
}
std::vector<float> convertToGrayscale(const RGBAImage& img) {
	std::vector<float> grayscale(img.width * img.height);

	for (int y = 0; y < img.height; ++y) {
		for (int x = 0; x < img.width; ++x) {
			int idx = y * img.width + x;

			const rgba& pixel = img.elements[idx];
			grayscale[idx] = static_cast<unsigned char>(
				0.299f * pixel.r + 0.587f * pixel.g + 0.114f * pixel.b
				);
		}
	}

	return grayscale;
}
void saveDisparityMap(const char* filename, const std::vector<float>& disp_vector, int width, int height)
{
	std::vector<unsigned char> output(width * height);
	for (size_t i = 0; i < disp_vector.size(); ++i) {
		{
			output[i] = disp_vector[i];
		}
	}
	stbi_write_png(filename, width, height, 1, output.data(), width);

}
void convertToPLY(std::vector<float> disparity_map, const char* filename)
{
	std::ofstream plyFile(filename);

	plyFile << "ply\n";
	plyFile << "format ascii 1.0\n";
	plyFile << "element vertex " << IMAGE_WIDTH*IMAGE_HEIGHT << "\n";
	plyFile << "property float x\n";
	plyFile << "property float y\n";
	plyFile << "property float z\n";
	plyFile << "property uchar red\n";
	plyFile << "property uchar green\n";
	plyFile << "property uchar blue\n";
	plyFile << "end_header\n";

	for (int i = 0; i < IMAGE_HEIGHT; i++)
	{
		for (int j = 0; j < IMAGE_WIDTH; j++)
		{
			float d = disparity_map[i * IMAGE_WIDTH + j];
			//if (d <= 0) continue;

			double Z = FOCAL_LENGTH * BASELINE / d;
			double X = -BASELINE * (2 * j + d) / (2 * d);
			double Y = BASELINE * i / d;

			plyFile << X << " " << Y << " " << Z << " " << d << " " << d << " " << d <<"\n";

		}
	}
	plyFile.close();
	std::cout << "done";

}
