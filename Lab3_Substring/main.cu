#include <iostream>
#include "cuda_runtime.h"
#include <cstdint>
#include <vector>
#include <string>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

#define BYTECAST(X) static_cast<uint8_t>(X)

struct SubstringSymbolPos {
    uint8_t symbol;
    uint32_t substringNum;
    uint32_t positionInSubstring;
    SubstringSymbolPos(uint8_t sym, uint32_t substr_num, uint32_t pos_in_substr) noexcept:
        symbol{sym}, substringNum{substr_num}, positionInSubstring{pos_in_substr} {}
};

enum SearchType {
    ANY, ALL_POS
};

__global__ void searchSubstrings(uint32_t bufferLength, uint32_t substringsCount, uint32_t minSearchLength,
                                 uint32_t maxSearchLength, SearchType searchType = SearchType::ANY) {

}


struct bytegen {
    __device__ uint8_t operator () (int idx)
    {
        thrust::default_random_engine randEng;
        //randEng.seed(8657863);
        thrust::uniform_int_distribution<uint8_t> uniDist(128, 150);
        randEng.discard(idx);
        return uniDist(randEng);
    }
};

enum Options {
    NONE,
    EXIT = 0,
    SUBSTR = 1,
    DEVICE_INFO = 2
};

int main() {
    setlocale(LC_ALL, "ru_RU.UTF-8");

    uint32_t searchStringNum;
    std::cout << "Введите количество строк:";
    std::cin >> searchStringNum;
    if (searchStringNum > 0) {
        auto searchStrings = std::vector<std::string>(searchStringNum);
        auto positions = std::vector<SubstringSymbolPos>();
        for (auto &s: searchStrings) {
            std::cout << "Введите строку:";
            std::cin >> s;
        }
        for (auto &s: searchStrings) {
            std::cout << s << std::endl;
        }
        for (uint32_t i = 0; i < searchStringNum; ++i) {
            uint32_t length = searchStrings[i].length();
            for (uint32_t j = 0; j < length; ++j) {
                positions.emplace_back(searchStrings[i][j], i, j);
            }
        }
        for (auto &p: positions) {
            auto[s, sn, pos] = p;
            std::cout << "symbol " << s << ", substringNum " << sn << ", positionInSubstring " << pos << std::endl;
        }
    }
    //uint8_t* stringBuffer, searchMatrix;
    uint32_t stringBufferSize;

    std::cout << "Введите размер буфера:";
    std::cin >> stringBufferSize;

    //Заполняем буфер случайными символами
    thrust::device_vector<uint8_t> d(stringBufferSize);
    thrust::transform(
            thrust::make_counting_iterator((uint32_t)0),
            thrust::make_counting_iterator(stringBufferSize),
            d.begin(),
            bytegen());
    //thrust::fill(d.begin(), d.begin() + stringBufferSize, bytegen);
    // copy a device_vector into an STL vector
    std::vector<uint8_t > stl_vector(d.size());
    thrust::copy(d.begin(), d.end(), stl_vector.begin());
    for (auto& sym: stl_vector) {
        std::wcout << sym << "|";
    }
    std::cout << std::endl;

    SubstringSymbolPos p(129, 45, 45);
    Options option = Options::NONE;
    bool main_cycle = true;
    while(main_cycle) {
        printf("Список действий:\n");
        printf("0 - выход из программы\n");
        printf("1 - сумма вектора\n");
        printf("2 - информация о GPU\n");
        printf("Выберите действие:");
        scanf_s("%d", &option);
        switch (option) {
            case Options::EXIT:
                main_cycle = false;
                break;
            case Options::SUBSTR: {
                    std::vector<SubstringSymbolPos> positions;
                    positions.push_back(p);
                }
                break;
            default:
                printf("Данной опции не существует. Попробуйте ещё раз.\n");
                break;
        }
    }

    return 0;
}
