#include <iostream>
#include "cuda_runtime.h"
#include <cstdint>
#include <vector>

struct SubstringSymbolPos {
    uint8_t symbol;
    uint32_t substringNum;
    uint32_t positionInSubstring;
    SubstringSymbolPos(uint8_t sym, uint32_t substr_num, uint32_t pos_in_substr) noexcept:
        symbol{sym}, substringNum{substr_num}, positionInSubstring{pos_in_substr} {}
};

enum Options {
    NONE,
    EXIT = 0,
    SUBSTR = 1,
    DEVICE_INFO = 2
};

int main() {
    setlocale(LC_ALL, "ru_RU.UTF-8");

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
