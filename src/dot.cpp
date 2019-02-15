#include <dot.h>
#include <functional>
#include <numeric>

float MultLocalList(std::list<float>& m1, std::list<float>& m2)
{
  return std::inner_product(m1.begin(), m1.end(), m2.begin(),0.0f);
}

float MultLocalVector(std::vector<float>& m1, std::vector<float>& m2)
{
	return std::inner_product(m1.begin(), m1.end(), m2.begin(), 0.0f);
}

float MultLocal(const float * m1, const float * m2, const int width)
{
    float result = 0.0f;
    for(int i = 0; i < width; i++)
    {
        result += m1[i] * m2[i];
    }
    return result;
}
float MultLocal2(const float * const m1, const float * const m2, const int width)
{
    float result = 0.0f;
    const int vectorSize = 2;
    for(int i = 0; i < width; i=i+vectorSize)
    {
        float result1 = m1[i] * m2[i];
        float result2 = m1[i+1] * m2[i+1];
        result += result1+result2;
    }
    const int start = (width/vectorSize)*vectorSize;
    for(int i = start; i<width;i++)
    {
        result += m1[i]*m2[i];
    }
    return result;
}
float MultLocal4 (const float *const m1, const float *const m2, const int width)
{
    float result = 0.0f;
    const int vectorSize = 4;
    for(int i = 0; i < width; i=i+vectorSize)
    {
        float result1 = m1[i] * m2[i];
        float result2 = m1[i+1] * m2[i+1];
        float result3 = m1[i+2] * m2[i+2];
        float result4 = m1[i+3] * m2[i+3];
        result += result1+result2+result3+result4;
    }
    const int start = (width/vectorSize)*vectorSize;
    for(int i = start; i<width;i++)
    {
        result += m1[i]*m2[i];
    }
    return result;
}


float MultLocal4Vect (const float *const m1, const float *const m2, const int width)
{
  float result1 = 0.0f;
  float result2 = 0.0f;
  float result3 = 0.0f;
  float result4 = 0.0f;
  const int vectorSize = 4;
  for(int i = 0; i < width; i=i+vectorSize)
    {
      result1 += m1[i] * m2[i];
      result2 += m1[i+1] * m2[i+1];
      result3 += m1[i+2] * m2[i+2];
      result4 += m1[i+3] * m2[i+3];
    }
  const int start = (width/vectorSize)*vectorSize;
  for(int i = start; i<width;i++)
    {
      result4 += m1[i]*m2[i];
    }
  return result1+result2+result3+result4;
}
float MultLocal8(const float * const m1, const float * const m2, const int width)
{
    float result = 0.0f;
    const int vectorSize = 8;
    for(int i = 0; i < width; i=i+vectorSize)
    {
        float result1 = m1[i] * m2[i];
        float result2 = m1[i+1] * m2[i+1];
        float result3 = m1[i+2] * m2[i+2];
        float result4 = m1[i+3] * m2[i+3];
        float result5 = m1[i+4] * m2[i+4];
        float result6 = m1[i+5] * m2[i+5];
        float result7 = m1[i+6] * m2[i+6];
        float result8 = m1[i+7] * m2[i+7];
        result += result1+result2+result3+result4+result5+result6+result7+result8;
    }
    const int start = (width/vectorSize)*vectorSize;
    for(int i = start; i<width;i++)
    {
        result += m1[i]*m2[i];
    }
    return result;
}
float MultLocal16(const float * const m1, const float * const m2, const int width)
{
    float result = 0.0f;
    const int vectorSize = 16;
    for(int i = 0; i < width; i=i+vectorSize)
    {
        float result1 = m1[i] * m2[i];
        float result2 = m1[i+1] * m2[i+1];
        float result3 = m1[i+2] * m2[i+2];
        float result4 = m1[i+3] * m2[i+3];
        float result5 = m1[i+4] * m2[i+4];
        float result6 = m1[i+5] * m2[i+5];
        float result7 = m1[i+6] * m2[i+6];
        float result8 = m1[i+7] * m2[i+7];
        float result9 = m1[i+8] * m2[i+8];
        float result10 = m1[i+9] * m2[i+9];
        float result11 = m1[i+10] * m2[i+10];
        float result12 = m1[i+11] * m2[i+11];
        float result13 = m1[i+12] * m2[i+12];
        float result14 = m1[i+13] * m2[i+13];
        float result15 = m1[i+14] * m2[i+14];
        float result16 = m1[i+15] * m2[i+15];
        result += result1+result2+result3+result4+result5+result6+result7+result8+result9+result10+result11+result12+result13+result14+result15+result16;
    }
    const int start = (width/vectorSize)*vectorSize;
    for(int i = start; i<width;i++)
    {
        result += m1[i]*m2[i];
    }
    return result;
}
float MultLocal32(const float * const m1, const float * const m2, const int width)
{
    float result = 0.0f;
    const int vectorSize = 16;
    for (int i = 0; i < width ; i = i + vectorSize)
    {
        float result1 = m1[i] * m2[i];
        float result2 = m1[i + 1] * m2[i + 1];
        float result3 = m1[i + 2] * m2[i + 2];
        float result4 = m1[i + 3] * m2[i + 3];
        float result5 = m1[i + 4] * m2[i + 4];
        float result6 = m1[i + 5] * m2[i + 5];
        float result7 = m1[i + 6] * m2[i + 6];
        float result8 = m1[i + 7] * m2[i + 7];
        float result9 = m1[i + 8] * m2[i + 8];
        float result10 = m1[i + 9] * m2[i + 9];
        float result11 = m1[i + 10] * m2[i + 10];
        float result12 = m1[i + 11] * m2[i + 11];
        float result13 = m1[i + 12] * m2[i + 12];
        float result14 = m1[i + 13] * m2[i + 13];
        float result15 = m1[i + 14] * m2[i + 14];
        float result16 = m1[i + 15] * m2[i + 15];
        float result17 = m1[i + 16] * m2[i + 16];
        float result18 = m1[i + 17] * m2[i + 17];
        float result19 = m1[i + 18] * m2[i + 18];
        float result20 = m1[i + 19] * m2[i + 19];
        float result21 = m1[i + 20] * m2[i + 20];
        float result22 = m1[i + 21] * m2[i + 21];
        float result23 = m1[i + 22] * m2[i + 22];
        float result24 = m1[i + 23] * m2[i + 23];
        float result25 = m1[i + 24] * m2[i + 24];
        float result26 = m1[i + 25] * m2[i + 25];
        float result27 = m1[i + 26] * m2[i + 26];
        float result28 = m1[i + 27] * m2[i + 27];
        float result29 = m1[i + 28] * m2[i + 28];
        float result30 = m1[i + 29] * m2[i + 29];
        float result31 = m1[i + 30] * m2[i + 30];
        float result32 = m1[i + 31] * m2[i + 31];
        result += result1 + result2 + result3 + result4 + result5 + result6 + result7 + result8 + result9 +
                  result10 + result11 + result12 + result13 + result14 + result15 + result16 + result17 + result18
                  + result19 +
                  result20 + result21 + result22 + result23 + result24 + result25 + result26 + result27 + result28
                  + result29 +
                  result30 + result31 + result32;
    }
    const int start = (width / vectorSize) * vectorSize;
    for (int i = start; i < width; i++)
    {
        result += m1[i] * m2[i];
    }
    return result;
}

float MultOpti(const float * const m1, const float * const m2, const int width)
{
	//Optimal SSE versions
#ifdef __AVX2__
	const int length8 = (width / 8) * 8;
	return dot256(m1, m2, length8) + MultLocal(m1 + length8, m2 + length8, width - length8);
#endif
#ifdef __SSE__
	const int length4 = (width / 4) * 4;
	return dot128(m1, m2, length4) + MultLocal(m1 + length4, m2 + length4, width - length4);
#endif
	return MultLocal(m1, m2, width);
}

#ifdef __SSE__

float dot128(const float *x1, const float *x2, size_t len) {
    assert(len % 4 == 0);
    __m128 sum = _mm_setzero_ps();
    if (len > 3) {
        size_t limit = len - 3;
        for (size_t i = 0; i < limit; i += 4) {
            __m128 v1 = _mm_loadu_ps(x1 + i);//equivalent to float[4] or Vec4
            __m128 v2 = _mm_loadu_ps(x2 + i);//equivalent to float[4] or Vec4
            sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));//equivalent to sum = sum + v1 * v2
            }
        }
    float buffer[4];
    _mm_storeu_ps(buffer, sum);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3];
}
#endif
#ifdef __SSE4_2__
// wmu insisted
float dot128dt(float *x1, float *x2, size_t len) {
    assert(len % 4 == 0);
    float sum = 0;
    if (len > 3) {
        size_t limit = len - 3;
        for (size_t i = 0; i < limit; i += 4) {
            __m128 v1 = _mm_loadu_ps(x1 + i);
            __m128 v2 = _mm_loadu_ps(x2 + i);
            sum += _mm_cvtss_f32(_mm_dp_ps(v1, v2, 0xf1));
            }
        }
    return sum;
}
#endif

#ifdef __FMA__
float dot128fma(float *x1, float *x2, size_t len) {
  assert(len % 4 == 0);
  __m128 sum = _mm_setzero_ps();
  if (len > 3) {
    size_t limit = len - 3;
    for (size_t i = 0; i < limit; i += 4) {
      __m128 v1 = _mm_loadu_ps(x1 + i);
      __m128 v2 = _mm_loadu_ps(x2 + i);
      sum = _mm_fmadd_ps(v1, v2, sum);
    }
  }
  float buffer[4];
  _mm_storeu_ps(buffer, sum);
  return buffer[0] + buffer[1] + buffer[2] + buffer[3];
}

#endif

#ifdef __AVX2__
float dot256(const float *x1, const float *x2, size_t len) {
  assert(len % 8 == 0);
  __m256 sum = _mm256_setzero_ps();
  if (len > 7) {
    size_t limit = len - 7;
    for (size_t i = 0; i < limit; i += 8) {
      __m256 v1 = _mm256_loadu_ps(x1 + i);
      __m256 v2 = _mm256_loadu_ps(x2 + i);
      sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
    }
  }
  float buffer[8];
  _mm256_storeu_ps(buffer, sum);
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];

}
#endif

#ifdef __FMA__

float dot256fma(float *x1, float *x2, size_t len) {
  assert(len % 8 == 0);
  __m256 sum = _mm256_setzero_ps();
  if (len > 7) {
    size_t limit = len - 7;
    for (size_t i = 0; i < limit; i += 8) {
      __m256 v1 = _mm256_loadu_ps(x1 + i);
      __m256 v2 = _mm256_loadu_ps(x2 + i);
      sum = _mm256_fmadd_ps(v1, v2, sum);
    }
  }
  float buffer[8];
  _mm256_storeu_ps(buffer, sum);
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];
}
#endif

#ifdef __AVX512F__

float dot512(float *x1, float *x2, size_t len) {
  assert(len % 16 == 0);
  __m512 sum = _mm512_setzero_ps();
  if (len > 15) {
    size_t limit = len - 15;
    for (size_t i = 0; i < limit; i += 16) {
      __m512 v1 = _mm512_loadu_ps(x1 + i);
      __m512 v2 = _mm512_loadu_ps(x2 + i);
      sum = _mm512_add_ps(sum, _mm512_mul_ps(v1, v2));
    }
  }
  float buffer[16];
  _mm512_storeu_ps(buffer, sum);
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7] + buffer[8] + buffer[9] + buffer[10] +
         buffer[11] + buffer[12] + buffer[13] + buffer[14] + buffer[15];
}

float dot512fma(float *x1, float *x2, size_t len) {
  assert(len % 16 == 0);
  __m512 sum = _mm512_setzero_ps();
  if (len > 15) {
    size_t limit = len - 15;
    for (size_t i = 0; i < limit; i += 16) {
      __m512 v1 = _mm512_loadu_ps(x1 + i);
      __m512 v2 = _mm512_loadu_ps(x2 + i);
      sum = _mm512_fmadd_ps(v1, v2, sum);
    }
  }
  float buffer[16];
  _mm512_storeu_ps(buffer, sum);
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7] + buffer[8] + buffer[9] + buffer[10] +
         buffer[11] + buffer[12] + buffer[13] + buffer[14] + buffer[15];
}
float dot512fma2(float *x1, float *x2, size_t len) {
  assert(len % 32 == 0);
  __m512 sum = _mm512_setzero_ps();
  if (len > 31) {
    size_t limit = len - 31;
    for (size_t i = 0; i < limit; i += 32) {

      __m512 v11 = _mm512_loadu_ps(x1 + i);
      __m512 v21 = _mm512_loadu_ps(x2 + i);
      __m512 v12 = _mm512_loadu_ps(x1 + i + 16);
      __m512 v22 = _mm512_loadu_ps(x2 + i + 16);
      sum = _mm512_fmadd_ps(v11, v21, sum);
      sum = _mm512_fmadd_ps(v12, v22, sum);
    }
  }
  float buffer[16];
  _mm512_storeu_ps(buffer, sum);
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7] + buffer[8] + buffer[9] + buffer[10] +
         buffer[11] + buffer[12] + buffer[13] + buffer[14] + buffer[15];
}

#endif