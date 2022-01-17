//
// Created by Vladislav on 17.01.2022.
//

#ifndef LAB3_SUBSTRING_SUBSTRINGSYMBOLPOS_HPP
#define LAB3_SUBSTRING_SUBSTRINGSYMBOLPOS_HPP
struct SubstringSymbolPos {
    char symbol;
    uint32_t substringNum;
    uint32_t positionInSubstring;
    SubstringSymbolPos(char sym, uint32_t substr_num, uint32_t pos_in_substr) noexcept:
    symbol{sym}, substringNum{substr_num}, positionInSubstring{pos_in_substr} {}
};
#endif //LAB3_SUBSTRING_SUBSTRINGSYMBOLPOS_HPP
