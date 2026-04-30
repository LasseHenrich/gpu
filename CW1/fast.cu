#include <random>
#include <iostream>

void init(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    std::mt19937 prng(2024);
    std::uniform_int_distribution<int32_t> distrib(-16, 16);

    for (auto i = 0; i < size; i++)
    {
        vec_a[i] = distrib(prng);
        vec_b[i] = distrib(prng);
    }

    for (auto i = 0; i < size * size; i++)
        mat[i] = distrib(prng);
}

void pretty_print(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    std::cout << "Vec A:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_a[i] << std::endl;

    std::cout << "Vec B:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_b[i] << std::endl;

    std::cout << "Matrix:" << std::endl;
    for (auto i = 0; i < size; i++)
    {
        for (auto j = 0; j < size; j++)
            std::cout << mat[i * size + j] << " ";

        std::cout << std::endl;
    }
}

int main()
{
    int32_t size = 32768;

    auto vec_a = (int32_t *)malloc(sizeof(int32_t) * size);
    auto vec_b = (int32_t *)malloc(sizeof(int32_t) * size);
    auto mat = (int32_t *)malloc(sizeof(int32_t *) * size * size);
    auto out = (int32_t *)malloc(sizeof(int32_t) * size);

    init(size, vec_a, vec_b, mat);

    pretty_print(size, vec_a, vec_b, mat);
}