# Лабораторная работа №3: Массовый поиск подстрок с использованием CUDA
***

## Постановка задачи:

Необходимо произвести поиск (установить факт наличия и, возможно, местоположение) множества подстрок различной длины в буфере данных. Поиск производится на полном восьмибитном алфавите.

Языки: __C++__, __CUDA__

Входные данные: Буфер размером размером 1 000 ... 10 000 000 значений.

Выходные данные: время вычисления + результат проверки корректности.

## Описание работы программы на CUDA:

Алгоритм распараллеливания заключается в том, что каждая нить будет производить поиск своего элемента буфера среди подстрок и убавлять соответствующее значение в результирующей матрице. После обработки всех символов входного буфера, каждый нулевой элемент рабочей матрицы r[i,j] = 0
соответствует подстроке Ni найденной во входном буфере начиная с позиции j.

Сначала создаётся результирующая матрица. Каждая её строка заполняется числом символов в соответствующей строке поиска:

```
    thrust::device_vector<uint32_t> resultMatrix(searchStringNum * stringBufferSize);
    uint32_t startPos, length;
    for (uint32_t i = 0; i < searchStringNum; ++i) {
        startPos = i * stringBufferSize;
        length = searchStrings[i].length();
        thrust::fill_n(thrust::device, resultMatrix.begin() + startPos, stringBufferSize, length);
    }
```

Далее производим вычитание элементов результирующей матрицы по алгоритму:

```
    auto rawBuffer = static_cast<const char*>(thrust::raw_pointer_cast(&stringBuffer[0]));
    auto rawPos = static_cast<SubstringSymbolPos*>(thrust::raw_pointer_cast(&devicePositions[0]));
    auto rawMatrix = static_cast<uint32_t*>(thrust::raw_pointer_cast(&resultMatrix[0]));
    searchSubstrings<<<stringBufferSize/blockSize, blockSize>>>(rawPos, positions.size(), rawBuffer, stringBufferSize, rawMatrix);
```
```
    __global__ void searchSubstrings(const SubstringSymbolPos* positions, uint32_t positionsSize, const char* stringBuffer,
                                 uint32_t stringBufferSize, uint32_t* resultMatrix) {
        auto index = blockDim.x * blockIdx.x + threadIdx.x;
        for (uint32_t i = 0; i < positionsSize; ++i) {
            if (stringBuffer[index] == positions[i].symbol) {
                auto resultIndex = positions[i].substringNum * stringBufferSize + index - positions[i].positionInSubstring;
                atomicSub(&resultMatrix[resultIndex], static_cast<uint32_t>(1));
            }
        }
    }
}
```

В конце происходит поиск индексов нулевых элементов, сортировка их по возрастанию и их запись в результирующий массив:

```
    int foundCount = thrust::count(resultMatrix.begin(), resultMatrix.end(), 0);
    thrust::device_vector<uint32_t> foundPos(foundCount);

    thrust::copy_if(
            thrust::device,
            thrust::make_counting_iterator((uint32_t)0),
            thrust::make_counting_iterator(stringBufferSize * searchStringNum),
            resultMatrix.begin(),
            foundPos.begin(),
            _1 == 0
    );
    thrust::sort(thrust::device, foundPos.begin(), foundPos.end());

    auto resultStlVector = std::vector<uint32_t>(foundCount);
    thrust::copy(foundPos.begin(), foundPos.end(), resultStlVector.begin());
```

Результатом поиска подстрок будет вектор индексов, из которых можно вычислить порядковый номер подстроки и её позицию в буфере.

## Описание работы программы на C++:

Для автоматического распараллеливания программы на C++ был применён API OpenMP:

```
    #pragma omp declare reduction (merge : std::vector<uint32_t> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    double start = omp_get_wtime();
#pragma omp parallel for collapse(2) default(none) shared(matrixSize, resultMatrix, stringBufferSize, positions, positionsSize, stringBuffer)
    for (uint32_t i = 0; i < stringBufferSize; ++i) {
        for (uint32_t j = 0; j < positionsSize; ++j) {
            if (stringBuffer[i] == positions[j].symbol) {
                auto resultIndex = positions[j].substringNum * stringBufferSize + i - positions[j].positionInSubstring;
#pragma omp atomic
                resultMatrix[resultIndex]--;
            }
        }
    }

    std::vector<uint32_t> result;
#pragma omp parallel for default(none) shared(matrixSize, resultMatrix) reduction(merge: result)
    for (uint32_t i = 0; i < matrixSize; ++i) {
        if (resultMatrix[i] == 0) result.push_back(i);
    }
    double end = omp_get_wtime();
```

Результатом поиска подстрок будет вектор индексов, из которых можно вычислить порядковый номер подстроки и её позицию в буфере.

## Пример работы программы:

Пример работы программы с количеством элементов вектора 10 240 000:

![Работа программы с количеством элементов вектора 10 240 000](https://github.com/Code5150/HPCProjects/blob/main/Lab3_Substring/img/l1_million_work.jpg)


## Результаты экспериментов:

Для замеров производился поиск 10 коротких строк (от 2 до 10 элементов) на сгенерированном массиве данных.

| Размер буфера     | 1 024    | 5 120    | 10 240   | 51 200      | 102 400     | 512 000      | 1 024 000    | 5 120 000    | 10 240 000   | 
| ----------------- | -------- | -------- | -------- | ----------- | ----------- | ------------ | ------------ | ------------ | ------------ | 
| Время на CPU, с   | 0,030776 | 0,000630 | 0,000704 | 0,002206    | 0,003372    | 0,015223     | 0,023903     | 0,155081     | 0,269473     |  
| Время на GPU, мс  | 0,595776 | 0,717824 | 0,721920 | 0,903168    | 1,090432    | 1,867776     | 2,884608     | 10,322016    | 19,206144    | 
| Ускорение, раз    |   51657  | 877,6525 | 975,1773 | 2442,513    | 3092,352    | 8150,335     | 8286,395     | 15024,29     | 14030,56     | 

График зависимости времени работы программы на __CPU__ от количества элементов вектора:

![График зависимости времени работы программы на CPU от количества элементов вектора](https://github.com/Code5150/HPCProjects/blob/main/Lab3_Substring/img/l1_cpu.jpg)

График зависимости времени работы программы на __GPU__ от количества элементов вектора:

![График зависимости времени работы программы на GPU от количества элементов вектора](https://github.com/Code5150/HPCProjects/blob/main/Lab3_Substring/img/l1_gpu.jpg)

График зависимости __ускорения__ от размера матрицы:

![График зависимости ускорения от размера количества элементов вектора](https://github.com/Code5150/HPCProjects/blob/main/Lab3_Substring/img/l1_speedup.jpg)

## Выводы:

1. Программа с использованием CUDA (GPU) работает, в зависимости от размера входных данных, в сотни и тысячи раз быстрее, чем на CPU;
2. С увеличением количества элементов буфера увеличивается и время работы программы. 