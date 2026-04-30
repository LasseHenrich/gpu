#include <random>
#include <iostream>

// ToDo: Should this initialization also be done on the GPU?
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

int main()
{
    int32_t size = 32768;

    auto vec_a = (int32_t *)malloc(sizeof(int32_t) * size);
    auto vec_b = (int32_t *)malloc(sizeof(int32_t) * size);
    auto mat = (int32_t *)malloc(sizeof(int32_t *) * size * size);
    auto out = (int32_t *)malloc(sizeof(int32_t) * size);

    init(size, vec_a, vec_b, mat);

    std::cout << "hello" << std::endl;
}