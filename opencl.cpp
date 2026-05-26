#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#ifdef _MSC_VER
#pragma comment(lib, "OpenCL.lib")
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
namespace fs = filesystem;
using Byte = unsigned char;
struct Image {
    int width = 0;
    int height = 0;
    int channels = 3;//ток ргб в проекте
    vector<Byte> data;

    int pixels() const { return width * height; }//кол-во пикселей
    int bytes() const { return width * height * channels; }//кол-во байт
};

enum class Filter { Invert, Median, Sobel };//класс фильтров

static const char* filterName(Filter f) {//функция вывода имени фильтра
    switch (f) {
        case Filter::Invert: return "invert";
        case Filter::Median: return "median";
        case Filter::Sobel:  return "sobel";
    }
    return "unknown";
}

static Byte clampToByte(int v) {//ограничение значения в диапазоне, чтобы в собеле не вылазить за предел
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<Byte>(v);
}

static Byte luminance(const Byte* p) {//читаем яркость пикселя чтоб в серый сделать
    return static_cast<Byte>(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
}

static Image loadPngRgb(const fs::path& path) {//загрузка картинки+конвертация в ргб
    int w = 0, h = 0, originalChannels = 0;
    Byte* raw = stbi_load(path.string().c_str(), &w, &h, &originalChannels, 3);
    if (!raw) throw runtime_error("Cannot load image: " + path.string());//ошибка если чет не так

    Image img;
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data.assign(raw, raw + w * h * 3);
    stbi_image_free(raw);
    return img;//возвращаем картинку в ргб
}

static void ensureOutputLike(const Image& input, Image& output) {//вывод такого же размера
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data.resize(input.data.size());
}

static void savePngRgb(const fs::path& path, const Image& img) {//сохранение в png
    fs::create_directories(path.parent_path());
    int ok = stbi_write_png(path.string().c_str(), img.width, img.height, img.channels,
                            img.data.data(), img.width * img.channels);
    if (!ok) throw runtime_error("Cannot save image: " + path.string());
}

template <typename Func>
static double measureAverageMs(Func&& func, int repeats) {//измерение времени выполнения функции(срзнач)
    using Clock = chrono::steady_clock;
    func(); // warm-up
    double totalMs = 0.0;
    for (int i = 0; i < repeats; ++i) {
        auto t0 = Clock::now();
        func();
        auto t1 = Clock::now();
        totalMs += chrono::duration<double, milli>(t1 - t0).count();
    }
    return totalMs / repeats;
}

// OpenCL

static const char* OPENCL_SOURCE = R"CLC(
__kernel void invert_rgb(
    __global const uchar* src,
    __global uchar* dst,
    const int w,
    const int h
) {
    //инверсия цветов
    int gid = get_global_id(0);
    int pixels = w * h;

    if (gid >= pixels) return;

    int i = gid * 3;
    //номер пикселя=gid, индекс байта=номер пикселя*кол-во каналов

    dst[i + 0] = (uchar)(255 - src[i + 0]);
    dst[i + 1] = (uchar)(255 - src[i + 1]);
    dst[i + 2] = (uchar)(255 - src[i + 2]);
}

uchar median9(uchar a[9]) {
    #define SWAP(i, j) do {\
        uchar x = a[i];\
        uchar y = a[j];\
        a[i] = min(x, y);\
        a[j] = max(x, y);\
    } while (0)

    SWAP(0,1); SWAP(3,4); SWAP(6,7);
    SWAP(1,2); SWAP(4,5); SWAP(7,8);
    SWAP(0,1); SWAP(3,4); SWAP(6,7);
    SWAP(0,3); SWAP(3,6); SWAP(0,3);
    SWAP(1,4); SWAP(4,7); SWAP(1,4);
    SWAP(2,5); SWAP(5,8); SWAP(2,5);
    SWAP(1,3); SWAP(5,7); SWAP(2,6);
    SWAP(4,6); SWAP(2,4); SWAP(2,3);
    SWAP(5,6);
    #undef SWAP
    return a[4];
}

__kernel void median_rgb(
    __global const uchar* src,
    __global uchar* dst,
    const int w,
    const int h
) {
    //медиана (не берем крайние пиксели птмчт нет соседей 3х3 не получится)
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= w || y >= h) return;

    int out = (y * w + x) * 3;

    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        dst[out + 0] = src[out + 0];
        dst[out + 1] = src[out + 1];
        dst[out + 2] = src[out + 2];
        return;
    }

    for (int c = 0; c < 3; ++c) {
        uchar values[9]; // 3x3
        int k = 0;
        // c=0—R отдельно каждый канал
        // c=1—G
        // c=2—B

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int idx = ((y + dy) * w + (x + dx)) * 3 + c;
                values[k++] = src[idx];
            }
        }

        dst[out + c] = median9(values);
    }
}

uchar gray_at(__global const uchar* src, int p) {
    //читаем яркость пикселя чтоб в серый сделать
    float r = (float)src[p + 0];
    float g = (float)src[p + 1];
    float b = (float)src[p + 2];
    return (uchar)(0.299f * r + 0.587f * g + 0.114f * b);
}

__kernel void sobel_rgb(
    __global const uchar* src,
    __global uchar* dst,
    const int w,
    const int h
) {
    //обнаружение границ собелем
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= w || y >= h) return;

    int out = (y * w + x) * 3;

    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        //все пиксели по границе в черный
        dst[out + 0] = 0;
        dst[out + 1] = 0;
        dst[out + 2] = 0;
        return;
    }
    //индексы 
    int a = ((y - 1) * w + (x - 1)) * 3;
    int b = ((y - 1) * w +  x ) * 3;
    int c = ((y - 1) * w + (x + 1)) * 3;
    int d = ( y* w + (x - 1)) * 3;
    int f = ( y* w + (x + 1)) * 3;
    int g = ((y + 1) * w + (x - 1)) * 3;
    int hh= ((y + 1) * w +  x) * 3;
    int i = ((y + 1) * w + (x + 1)) * 3;
    //яркости
    int A = gray_at(src, a);
    int B = gray_at(src, b);
    int C = gray_at(src, c);
    int D = gray_at(src, d);
    int F = gray_at(src, f);
    int G = gray_at(src, g);
    int H = gray_at(src, hh);
    int I = gray_at(src, i);

    //формула вертикального и горизонтального градиента
    int gx = (C + 2 * F + I) - (A + 2 * D + G);
    int gy = (A + 2 * B + C) - (G + 2 * H + I);

    //сила градиента=корень из суммы квадратов горизонтального и вертикального градиента
    int mag = (int)sqrt((float)(gx * gx + gy * gy));
    uchar v = (uchar)min(255, mag);
    //если сверху светло снизу темно то будет сильный градиент и наоборот, если одинаково то градиента нет

    dst[out + 0] = v;
    dst[out + 1] = v;
    dst[out + 2] = v;
}
)CLC";

static void clCheck(cl_int err, const char* what) {//проверка ошибок OpenCL после каждого вызова
    if (err != CL_SUCCESS) {
        throw runtime_error(string("OpenCL error in ") + what + ": " + to_string(err));
    }
}

struct OpenCLRuntime {//все объекты OpenCL которые нужны для запуска kernel
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;//очередь команд на устройство
    cl_program program = nullptr;//скомпилированная строка OPENCL_SOURCE
    cl_device_id device = nullptr;//GPU или другое OpenCL-устройство

    cl_kernel invert = nullptr;//готовые функции из OPENCL_SOURCE
    cl_kernel median = nullptr;
    cl_kernel sobel = nullptr;

    bool initialized = false;

    ~OpenCLRuntime() {//освобождаем OpenCL-ресурсы
        if (invert) clReleaseKernel(invert);
        if (median) clReleaseKernel(median);
        if (sobel)  clReleaseKernel(sobel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    void init() {//инициализация OpenCL один раз перед первым запуском
        if (initialized) return;

        cl_uint platformCount = 0;
        clCheck(clGetPlatformIDs(0, nullptr, &platformCount), "clGetPlatformIDs count");

        if (platformCount == 0) {
            throw runtime_error("OpenCL platforms not found. Install GPU/CPU OpenCL runtime driver.");
        }

        vector<cl_platform_id> platforms(platformCount);
        clCheck(clGetPlatformIDs(platformCount, platforms.data(), nullptr), "clGetPlatformIDs list");

        cl_int lastErr = CL_DEVICE_NOT_FOUND;

        for (cl_platform_id platform : platforms) {//сначала ищем GPU, если нет—default устройство
            lastErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
            if (lastErr == CL_SUCCESS) break;

            lastErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
            if (lastErr == CL_SUCCESS) break;
        }

        clCheck(lastErr, "clGetDeviceIDs");

        cl_int err = CL_SUCCESS;

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);//контекст связывает программу с устройством
        clCheck(err, "clCreateContext");

        queue = clCreateCommandQueue(context, device, 0, &err);//через очередь отправляем копирование данных и запуск kernel
        clCheck(err, "clCreateCommandQueue");

        const char* sources[] = { OPENCL_SOURCE };
        program = clCreateProgramWithSource(context, 1, sources, nullptr, &err);//создаем программу из строки с kernel'ами
        clCheck(err, "clCreateProgramWithSource");

        err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);//компиляция kernel'ов под конкретное устройство
        if (err != CL_SUCCESS) {
            size_t logSize = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

            vector<char> log(logSize + 1);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);

            throw runtime_error(string("OpenCL build failed:\n") + log.data());
        }

        invert = clCreateKernel(program, "invert_rgb", &err);//достаем kernel по имени функции из OPENCL_SOURCE
        clCheck(err, "clCreateKernel invert_rgb");

        median = clCreateKernel(program, "median_rgb", &err);
        clCheck(err, "clCreateKernel median_rgb");

        sobel = clCreateKernel(program, "sobel_rgb", &err);
        clCheck(err, "clCreateKernel sobel_rgb");

        initialized = true;
    }

    void runKernel(const Image& input, Image& output, cl_kernel kernel, bool twoDimensional) {//общий запуск любого OpenCL-фильтра
        init();

        if (kernel == nullptr) {
            throw runtime_error("OpenCL kernel is null. Initialization failed or wrapper passed kernel before init.");
        }

        ensureOutputLike(input, output);

        const int bytes = input.bytes();
        const int w = input.width;
        const int h = input.height;

        cl_int err = CL_SUCCESS;

        cl_mem dSrc = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err);//буфер входной картинки на устройстве
        clCheck(err, "clCreateBuffer dSrc");

        cl_mem dDst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);//буфер результата на устройстве
        clCheck(err, "clCreateBuffer dDst");

        //копируем входные пиксели CPU в GPU
        clCheck(clEnqueueWriteBuffer(queue, dSrc, CL_TRUE, 0, bytes, input.data.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");

        //передаем аргументы в kernel: src, dst, width, height
        clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dSrc), "clSetKernelArg 0");
        clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dDst), "clSetKernelArg 1");
        clCheck(clSetKernelArg(kernel, 2, sizeof(int), &w), "clSetKernelArg 2");
        clCheck(clSetKernelArg(kernel, 3, sizeof(int), &h), "clSetKernelArg 3");

        if (twoDimensional) {
            //median и sobel удобнее запускать по сетке width x height
            size_t global[2] = { static_cast<size_t>(w), static_cast<size_t>(h) };
            clCheck(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel 2D");
        } else {
            //invert идет одним пикселем
            size_t global = static_cast<size_t>(input.pixels());
            clCheck(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel 1D");
        }

        clCheck(clFinish(queue), "clFinish");

        clCheck(clEnqueueReadBuffer(queue, dDst, CL_TRUE, 0, bytes, output.data.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");

        //временные буферы создаются на каждый запуск и сразу освобождаются
        clReleaseMemObject(dSrc);
        clReleaseMemObject(dDst);
    }
};

static void invertOpenCL(OpenCLRuntime& cl, const Image& input, Image& output) {
    cl.init();
    cl.runKernel(input, output, cl.invert, false);
}

static void medianOpenCL(OpenCLRuntime& cl, const Image& input, Image& output) {
    cl.init();
    cl.runKernel(input, output, cl.median, true);
}

static void sobelOpenCL(OpenCLRuntime& cl, const Image& input, Image& output) {
    cl.init();
    cl.runKernel(input, output, cl.sobel, true);
}



static vector<fs::path> collectInputImages(const fs::path& inputPath) {//сбор входных картинок с автопоиском и сортировкой по имени
    vector<fs::path> files;

    if (fs::is_regular_file(inputPath)) {
        files.push_back(inputPath);
        return files;
    }

    if (!fs::is_directory(inputPath)) {
        throw runtime_error("Input path is neither file nor directory: " + inputPath.string());
    }

    for (const auto& entry : fs::directory_iterator(inputPath)) {
        if (!entry.is_regular_file()) continue;
        fs::path p = entry.path();
        string ext = p.extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(tolower(c));
        });
        if (ext == ".png") files.push_back(p);
    }

    auto firstNumber = [](const fs::path& path) {
        int number = 0;
        bool found = false;

        for (char ch : path.stem().string()) {
            if (ch < '0' || ch > '9') {
                if (found) break;
                continue;
            }
            found = true;
            number = number * 10 + (ch - '0');
        }

        return number;
    };

    sort(files.begin(), files.end(), [&](const fs::path& a, const fs::path& b) {
        const int numberA = firstNumber(a);
        const int numberB = firstNumber(b);
        if (numberA != numberB) return numberA < numberB;
        return a.filename().string() < b.filename().string();
    });
    if (files.empty()) throw runtime_error("No PNG files found in: " + inputPath.string());
    return files;
}

static void printUsage() {//инструкция
    cout
        << "Usage:\n"
        << "  opencl.exe [input_file_or_dir] [output_dir] [repeats]\n\n"
        << "Defaults:\n"
        << "  input_file_or_dir = \"images\"\n"
        << "  output_dir        = \"resultati\"\n"
        << "  repeats           = 10\n\n";
}

static void applyFilter(Filter filter, OpenCLRuntime& cl, const Image& input, Image& output) {//выбор нужного фильтра
    if (filter == Filter::Invert) invertOpenCL(cl, input, output);
    if (filter == Filter::Median) medianOpenCL(cl, input, output);
    if (filter == Filter::Sobel)  sobelOpenCL(cl, input, output);
}

int main(int argc, char** argv) {//кол-во аргументов сами аргументы
    try {
        setlocale(LC_ALL, "");

        fs::path inputPath = (argc >= 2) ? fs::path(argv[1]) : fs::path("images");//если первый аргумент есть то он путь к картинке иначе папка images
        fs::path outputDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("resultati");//если второй аргумент есть то он папка для сохранения результатов иначе папка resultati
        int repeats = (argc >= 4) ? max(1, stoi(argv[3])) : 10;//если третий аргумент есть то он кол-во повторов для измерения времени иначе 10

        if (argc >= 2) {//если есть аргумент то проверяем не -h и не --help и выводим инструкцию если это так
            string arg1 = argv[1];
            if (arg1 == "-h" || arg1 == "--help") {
                printUsage();
                return 0;
            }
        }

        vector<fs::path> inputs = collectInputImages(inputPath);

        fs::create_directories(outputDir);

        OpenCLRuntime opencl;
        cout << "image | filter | avg_time_ms | iterations_kol-vo\n";

        const vector<Filter> filters = { Filter::Invert, Filter::Median, Filter::Sobel };

        for (const fs::path& file : inputs) {
            Image input = loadPngRgb(file);
            Image output = input;

            for (Filter filter : filters) {
                const double avgMs = measureAverageMs([&]() {
                    applyFilter(filter, opencl, input, output);
                }, repeats);

                fs::path outPath = outputDir /
                    (file.stem().string() + "_" + filterName(filter) + "_opencl" + ".png");
                savePngRgb(outPath, output);

                cout << file.filename().string()
                          << " | " << filterName(filter)
                          << " | " << fixed << setprecision(2) << avgMs << " ms"
                          << " | " << repeats << "\n";
            }
        }

        cout << "done\n";
        return 0;
    }
    catch (const exception& e) {
        cerr << "\nERROR: " << e.what() << "\n\n";
        printUsage();
        return 1;
    }
}
