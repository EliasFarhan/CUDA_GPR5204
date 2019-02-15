//
// Created by efarhan on 22.12.18.
//

#ifndef GPR5204_MISC_H
#define GPR5204_MISC_H

void Prefill(float* m, const int size)
{
    for(int i = 0; i<size; i++)
    {
        m[i] = static_cast<float>(rand()%1000)/100.0f;
    }
}

#endif //GPR5204_MISC_H
