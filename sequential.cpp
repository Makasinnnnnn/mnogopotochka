#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
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

//последовательная

static void invertSequential(const Image& input, Image& output) {//инверсия цветов
    ensureOutputLike(input, output);

    const int n = input.bytes();
    for (int i = 0; i < n; ++i) {
        output.data[i] = static_cast<Byte>(255 - input.data[i]);
    }
}

static void medianSequential(const Image& input, Image& output) {//медиана (не берем крайние пиксели птмчт нет соседей 3х3 не получится)
    ensureOutputLike(input, output);
    output.data = input.data;

    const int w = input.width;
    const int h = input.height;
    const int ch = input.channels;

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const int center = (y * w + x) * ch;
//номер пикселя = y * width + x
//индекс байта = номер пикселя * кол-во каналов

            for (int c = 0; c < 3; ++c) {
                Byte values[9];//3x3
                int k = 0;
//c=0—R отдельно каждый канал
//c=1—G
//c=2—B

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int idx = ((y + dy) * w + (x + dx)) * ch + c;
                        values[k++] = input.data[idx];
                    }
                }

                nth_element(values, values + 4, values + 9);//быстрый поиск медианы чтобы прост не сортировать весь массив только 5 элемент(индекс 4) будет медианой
                output.data[center + c] = values[4];
            }
        }
    }
}

static void sobelSequential(const Image& input, Image& output) {
    ensureOutputLike(input, output);
    fill(output.data.begin(), output.data.end(), 0);//все пиксели в черный (границы)

    const int w = input.width;
    const int h = input.height;
    const int ch = input.channels;
    
    for (int y = 1; y < h - 1; ++y) {//8 соседей
        for (int x = 1; x < w - 1; ++x) {
            const int a = ((y - 1) * w + (x - 1)) * ch;
            const int b = ((y - 1) * w +  x     ) * ch;
            const int c = ((y - 1) * w + (x + 1)) * ch;
            const int d = ( y      * w + (x - 1)) * ch;
            const int f = ( y      * w + (x + 1)) * ch;
            const int g = ((y + 1) * w + (x - 1)) * ch;
            const int hh= ((y + 1) * w +  x     ) * ch;
            const int i = ((y + 1) * w + (x + 1)) * ch;

            const int A = luminance(&input.data[a]);//каждый в яркость
            const int B = luminance(&input.data[b]);
            const int C = luminance(&input.data[c]);
            const int D = luminance(&input.data[d]);
            const int F = luminance(&input.data[f]);
            const int G = luminance(&input.data[g]);
            const int H = luminance(&input.data[hh]);
            const int I = luminance(&input.data[i]);

//формула вертикального и горизонтального градиента
            const int gx = (C + 2 * F + I) - (A + 2 * D + G);
            const int gy = (A + 2 * B + C) - (G + 2 * H + I);

            const int value = static_cast<int>(sqrt(static_cast<double>(gx * gx + gy * gy)));//сила градиента = корень из суммы квадратов горизонтального и вертикального градиента
            const Byte v = clampToByte(value);
//если сверху светло снизу темно то будет сильный градиент и наоборот, если одинаково то градиента нет

            const int out = (y * w + x) * ch;
            output.data[out + 0] = v;
            output.data[out + 1] = v;
            output.data[out + 2] = v;
        }
    }
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
        << "  sequential.exe [input_file_or_dir] [output_dir] [repeats]\n\n"
        << "Defaults:\n"
        << "  input_file_or_dir = \"images\"\n"
        << "  output_dir        = \"resultati\"\n"
        << "  repeats           = 10\n\n";
}

static void applyFilter(Filter filter,  const Image& input, Image& output) {//выбор нужного фильтра
    if (filter == Filter::Invert) invertSequential(input, output);
    if (filter == Filter::Median) medianSequential(input, output);
    if (filter == Filter::Sobel)  sobelSequential(input, output);
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

        cout << "image | filter | avg_time_ms | iterations_kol-vo\n";

        const vector<Filter> filters = { Filter::Invert, Filter::Median, Filter::Sobel };

        for (const fs::path& file : inputs) {
            Image input = loadPngRgb(file);
            Image output = input;

            for (Filter filter : filters) {
                const double avgMs = measureAverageMs([&]() {
                    applyFilter(filter,  input, output);
                }, repeats);

                fs::path outPath = outputDir /
                    (file.stem().string() + "_" + filterName(filter) + "_sequential" + ".png");
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
