//
// Created by Vladislav on 17.01.2022.
//

#ifndef LAB3_SUBSTRING_CPUSUBSTRING_H
#define LAB3_SUBSTRING_CPUSUBSTRING_H
#include <vector>
#include <cstdint>
#include "SubstringSymbolPos.hpp"
std::vector<uint32_t> cpuSearchSubstrings(const SubstringSymbolPos* positions, uint32_t positionsSize, const char* stringBuffer,
                                          uint32_t stringBufferSize, uint32_t* resultMatrix, uint32_t matrixSize);
#endif //LAB3_SUBSTRING_CPUSUBSTRING_H
