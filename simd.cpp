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

#include <immintrin.h>

//SIMD 

static void invertSIMD(const Image& input, Image& output) {//инверсия цветов
    ensureOutputLike(input, output);//готовим выход 
//m128i 16 байт целые числа 
    const int n = input.bytes();//кол-во байт
    const __m128i mask = _mm_set1_epi8(static_cast<char>(0xFF));//симд регистр из 16 байт все по 255

    int i = 0;
    for (; i + 16 <= n; i += 16) {//идем по 16 байт 
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input.data[i]));//невыровненная загрузка безопасна даже для не кратных 16
        const __m128i r = _mm_xor_si128(v, mask); // сразу инвертируем через исключающее ИЛИ с 255 (0xFF) и сохраняем результат 16 байт 
        //xor переворот битов 
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&output.data[i]), r);
    }

    for (; i < n; ++i) {
        output.data[i] = static_cast<Byte>(255 - input.data[i]);
    }
}

static void splitRgbToPlanes(const Image& input, vector<Byte>& r, vector<Byte>& g, vector<Byte>& b) {//делим на р г и б
    const int n = input.pixels();
    r.resize(n);
    g.resize(n);
    b.resize(n);

    for (int i = 0; i < n; ++i) {
        r[i] = input.data[i * 3 + 0];
        g[i] = input.data[i * 3 + 1];
        b[i] = input.data[i * 3 + 2];
    }
}

static void mergePlanesToRgb(const vector<Byte>& r, const vector<Byte>& g, const vector<Byte>& b, Image& output) {//обратно мерджим в ргб
    const int n = output.pixels();

    for (int i = 0; i < n; ++i) {
        output.data[i * 3 + 0] = r[i];
        output.data[i * 3 + 1] = g[i];
        output.data[i * 3 + 2] = b[i];
    }
}

static void medianPlaneSIMD(const vector<Byte>& input, vector<Byte>& output, int w, int h) {//медиана (не берем крайние пиксели птмчт нет соседей 3х3 не получится)
    output = input;
//медиана только для одного канала =>1 байт/пиксель
    for (int y = 1; y < h - 1; ++y) {
        int x = 1;

        for (; x + 16 <= w - 1; x += 16) {//16 писелей соседей
            __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input[(y - 1) * w + (x - 1)]));
            __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input[(y - 1) * w +  x]));
            __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input[(y - 1) * w + (x + 1)]));
            __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input[ y * w + (x - 1)]));
            __m128i v4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input[ y* w +  x]));
            __m128i v5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input[ y * w + (x + 1)]));
            __m128i v6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input[(y + 1) * w + (x - 1)]));
            __m128i v7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input[(y + 1) * w +  x ]));
            __m128i v8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input[(y + 1) * w + (x + 1)]));
//if (a>b) swap (a and b) ток для всех 16 байт
            #define SWAP_MIN_MAX(a, b) do {\
                __m128i mn = _mm_min_epu8((a), (b));\
                __m128i mx = _mm_max_epu8((a), (b));\
                (a) = mn;\
                (b) = mx;\
            } while (0)
//свапы чтоб найти медиану для 16 пикселей (16 медиан параллельно) 
            SWAP_MIN_MAX(v0, v1); SWAP_MIN_MAX(v3, v4); SWAP_MIN_MAX(v6, v7);
            SWAP_MIN_MAX(v1, v2); SWAP_MIN_MAX(v4, v5); SWAP_MIN_MAX(v7, v8);
            SWAP_MIN_MAX(v0, v1); SWAP_MIN_MAX(v3, v4); SWAP_MIN_MAX(v6, v7);
            SWAP_MIN_MAX(v0, v3); SWAP_MIN_MAX(v3, v6); SWAP_MIN_MAX(v0, v3);
            SWAP_MIN_MAX(v1, v4); SWAP_MIN_MAX(v4, v7); SWAP_MIN_MAX(v1, v4);
            SWAP_MIN_MAX(v2, v5); SWAP_MIN_MAX(v5, v8); SWAP_MIN_MAX(v2, v5);
            SWAP_MIN_MAX(v1, v3); SWAP_MIN_MAX(v5, v7); SWAP_MIN_MAX(v2, v6);
            SWAP_MIN_MAX(v4, v6); SWAP_MIN_MAX(v2, v4); SWAP_MIN_MAX(v2, v3);
            SWAP_MIN_MAX(v5, v6);
            #undef SWAP_MIN_MAX
//сохраненние 16 медиан 
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&output[y * w + x]), v4);
        }
//если осталось меньше 16 пикселей то дефолтным способом до конца строки
        for (; x < w - 1; ++x){
            Byte values[9];//3x3
            int k = 0;

            for (int dy = -1; dy <= 1; ++dy){
                for (int dx = -1; dx <= 1; ++dx){
                    values[k++] = input[(y + dy) * w + (x + dx)];
                }
            }

            nth_element(values, values + 4, values + 9);//быстрый поиск медианы чтобы прост не сортировать весь массив только 5 элемент(индекс 4) будет медианой
            output[y * w + x] = values[4];
        }
    }
}

static void medianSIMD(const Image& input, Image& output) {//c=0—R, c=1—G, c=2—B отдельно каждый канал
    ensureOutputLike(input, output);

    vector<Byte> r, g, b;
    vector<Byte> ro, go, bo;//обработанные 

    splitRgbToPlanes(input, r, g, b);

    medianPlaneSIMD(r, ro, input.width, input.height);
    medianPlaneSIMD(g, go, input.width, input.height);
    medianPlaneSIMD(b, bo, input.width, input.height);

    mergePlanesToRgb(ro, go, bo, output);
}

static void rgbToGrayScalar(const Image& input, vector<Byte>& gray) {//делаем серое изображение
    const int n = input.pixels();
    gray.resize(n);

    for (int i = 0; i < n; ++i) {
        gray[i] = luminance(&input.data[i * 3]);//каждый в яркость
    }
}

static void sobelSIMD(const Image& input, Image& output) {//обнаружение границ 
    ensureOutputLike(input, output);
    fill(output.data.begin(), output.data.end(), 0);//все пиксели в черный (границы)

    const int w = input.width;
    const int h = input.height;

    vector<Byte> gray;
    vector<Byte> edge(input.pixels(), 0);
    rgbToGrayScalar(input, gray);

    for (int y = 1; y < h - 1; ++y) {//8 соседей/идем по строкам кроме границ
        int x = 1;

        for (; x + 8 <= w - 1; x += 8) {
            //epi64 грузит 8 байт epi16 расширяет 8 байт в 8 16-битных чисел 
            //8 значений uint8
            //8 значений int16
            const __m128i A = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&gray[(y - 1) * w + (x - 1)])));
            const __m128i B = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&gray[(y - 1) * w +  x])));
            const __m128i C = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&gray[(y - 1) * w + (x + 1)])));
            const __m128i D = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&gray[ y* w + (x - 1)])));
            const __m128i F = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&gray[ y* w + (x + 1)])));
            const __m128i G = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&gray[(y + 1) * w + (x - 1)])));
            const __m128i H = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&gray[(y + 1) * w +  x])));
            const __m128i I = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&gray[(y + 1) * w + (x + 1)])));

//формула вертикального и горизонтального градиента
            const __m128i right = _mm_add_epi16(C, _mm_add_epi16(_mm_slli_epi16(F, 1), I));
            const __m128i left  = _mm_add_epi16(A, _mm_add_epi16(_mm_slli_epi16(D, 1), G));
            const __m128i gx16  = _mm_sub_epi16(right, left);//сравниваем разницу для гарницы 
//gx16 и gy16 содержат 8 значений типа int16
//Для sqrt нужно перейти к float
//Сначала считаем первые 4 пикселя из этих 8
            const __m128i top    = _mm_add_epi16(A, _mm_add_epi16(_mm_slli_epi16(B, 1), C));
            const __m128i bottom = _mm_add_epi16(G, _mm_add_epi16(_mm_slli_epi16(H, 1), I));
            const __m128i gy16   = _mm_sub_epi16(top, bottom);//тоже самое 
//gx16 и gy16 содержат 8 значений типа int16
//Для sqrt нужно перейти к float
//Сначала считаем первые 4 пикселя из этих 8
            const __m128 gxLo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(gx16));
            const __m128 gyLo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(gy16));
//сила границы  это корень из суммы квадратов горизонтального и вертикального градиента
            const __m128 magLo = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(gxLo, gxLo), _mm_mul_ps(gyLo, gyLo)));
//обратно в инт и упаковываем в 8 бит
            const __m128i iLo = _mm_cvtps_epi32(magLo);
//Теперь берём вторые 4 пикселя
//Для этого сдвигаем SIMD-регистр на 8 байт
//потому что первые 4 значения int16 занимают  4*2=8 байт
            const __m128i gxHi16 = _mm_srli_si128(gx16, 8);
            const __m128i gyHi16 = _mm_srli_si128(gy16, 8);
//сила границы для других 4 пикселей
            const __m128 gxHi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(gxHi16));
            const __m128 gyHi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(gyHi16));
            const __m128 magHi = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(gxHi, gxHi), _mm_mul_ps(gyHi, gyHi)));
            const __m128i iHi = _mm_cvtps_epi32(magHi);
//Упаковываем 8 результатов обратно
//сначала int32 в int16, потом int16 в unsigned char
//_mm_packus_epi16 упаковывает 16-битные числа в 8-битные(значения меньше 0 становятся 0, а больше 255 становятся 255)
            const __m128i packed16 = _mm_packs_epi32(iLo, iHi);
            const __m128i packed8  = _mm_packus_epi16(packed16, packed16);

            _mm_storel_epi64(reinterpret_cast<__m128i*>(&edge[y * w + x]), packed8);
        }

        for (; x < w - 1; ++x) {
            const int A = gray[(y - 1) * w + (x - 1)];
            const int B = gray[(y - 1) * w +  x];
            const int C = gray[(y - 1) * w + (x + 1)];
            const int D = gray[ y * w + (x - 1)];
            const int F = gray[ y * w + (x + 1)];
            const int G = gray[(y + 1) * w + (x - 1)];
            const int H = gray[(y + 1) * w +  x];
            const int I = gray[(y + 1) * w + (x + 1)];

//формула вертикального и горизонтального градиента
            const int gx = (C + 2 * F + I) - (A + 2 * D + G);
            const int gy = (A + 2 * B + C) - (G + 2 * H + I);
            const int value = static_cast<int>(sqrt(static_cast<double>(gx * gx + gy * gy)));//сила градиента = корень из суммы квадратов горизонтального и вертикального градиента

            edge[y * w + x] = clampToByte(value);
//если сверху светло снизу темно то будет сильный градиент и наоборот, если одинаково то градиента нет
        }
    }

    for (int i = 0; i < input.pixels(); ++i) {
        output.data[i * 3 + 0] = edge[i];
        output.data[i * 3 + 1] = edge[i];
        output.data[i * 3 + 2] = edge[i];
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
        << "  simd.exe [input_file_or_dir] [output_dir] [repeats]\n\n"
        << "Defaults:\n"
        << "  input_file_or_dir = \"images\"\n"
        << "  output_dir        = \"resultati\"\n"
        << "  repeats           = 10\n\n";
}

static void applyFilter(Filter filter,  const Image& input, Image& output) {//выбор нужного фильтра
    if (filter == Filter::Invert) invertSIMD(input, output);
    if (filter == Filter::Median) medianSIMD(input, output);
    if (filter == Filter::Sobel)  sobelSIMD(input, output);
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
                    (file.stem().string() + "_" + filterName(filter) + "_simd" + ".png");
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
