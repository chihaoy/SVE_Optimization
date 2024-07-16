#include "bl_config.h"
#include "bl_dgemm_kernel.h"

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

// X macros to make my life easier
#define REPEAT_4(X) X(0) X(1) X(2) X(3)
#define REPEAT_1_4(X) X(1) X(2) X(3)
#define REPEAT_8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define REPEAT_1_8(X) X(1) X(2) X(3) X(4) X(5) X(6) X(7)

//
// C-based micorkernel
//
void bl_dgemm_ukr(int k,
                  int m,
                  int n,
                  const double *restrict a,
                  const double *restrict b,
                  double *restrict c,
                  unsigned long long ldc,
                  aux_t *data)
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < n; ++j )
        { 
            for ( i = 0; i < m; ++i )
            { 
	      // ldc is used here because a[] and b[] are not packed by the
	      // starter code
	      // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
	      //
	      c( i, j, ldc ) += a( l, i, m) * b( l, j, n );   
            }
        }
    }
}

void bl_dgemm_ukr_sve_4x4(int k,
                      int m,
                      int n,
                      const double *restrict a,
                      const double *restrict b,
                      double *restrict c,
                      unsigned long long ldc,
                      aux_t *data)
{
    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x;
    register float64_t aval;

    svbool_t npred = svwhilelt_b64_u64(0, n);
    svbool_t npred_m_gt1 = svmov_z(npred, svdup_b64(m > 1));
    svbool_t npred_m_gt2 = svmov_z(npred, svdup_b64(m > 2));
    svbool_t npred_m_gt3 = svmov_z(npred, svdup_b64(m > 3));

    c0x = svld1_f64(npred, c);
    c1x = svld1_f64(npred_m_gt1, c + ldc);
    c2x = svld1_f64(npred_m_gt2, c + ldc * 2);
    c3x = svld1_f64(npred_m_gt3, c + ldc * 3);

    for (int kk = 0; kk < k; kk++)
    {
        bx = svld1_f64(npred, b + kk * n);

        aval = *(a + kk * m);
        ax = svdup_f64(aval);
        c0x = svmla_f64_m(npred, c0x, bx, ax);

        aval = *(a + kk * m + 1);
        ax = svdup_f64(aval);
        c1x = svmla_f64_m(npred_m_gt1, c1x, bx, ax);

        aval = *(a + kk * m + 2);
        ax = svdup_f64(aval);
        c2x = svmla_f64_m(npred_m_gt2, c2x, bx, ax);

        aval = *(a + kk * m + 3);
        ax = svdup_f64(aval);
        c3x = svmla_f64_m(npred_m_gt3, c3x, bx, ax);
    }

    svst1_f64(npred, c, c0x);
    svst1_f64(npred_m_gt1, c + ldc, c1x);
    svst1_f64(npred_m_gt2, c + ldc * 2, c2x);
    svst1_f64(npred_m_gt3, c + ldc * 3, c3x);
}

void bl_dgemm_ukr_sve_4x4_alternative(
    int k,
    int m,
    int n,
    const double *restrict a,
    const double *restrict b,
    double *restrict c,
    unsigned long long ldc,
    aux_t *data)
{
    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x;
    register svfloat64_t a0101, a2323;

    svbool_t mpred = svwhilelt_b64_u64(0, m);
    svbool_t npred = svwhilelt_b64_u64(0, n);
    svbool_t npred_m_gt1 = svmov_z(npred, svdup_b64(m > 1));
    svbool_t npred_m_gt2 = svmov_z(npred, svdup_b64(m > 2));
    svbool_t npred_m_gt3 = svmov_z(npred, svdup_b64(m > 3));

    c0x = svld1_f64(npred, c);
    c1x = svld1_f64(npred_m_gt1, c + ldc);
    c2x = svld1_f64(npred_m_gt2, c + ldc * 2);
    c3x = svld1_f64(npred_m_gt3, c + ldc * 3);

    for (int kk = 0; kk < k; kk++)
    {
        ax = svld1_f64(mpred, a + kk * m);
        bx = svld1_f64(npred, b + kk * n);

        a0101 = svdupq_lane_f64(ax, 0);
        a2323 = svdupq_lane_f64(ax, 1);

        c0x = svmla_lane_f64(c0x, bx, a0101, 0);
        c1x = svmla_lane_f64(c1x, bx, a0101, 1);
        c2x = svmla_lane_f64(c2x, bx, a2323, 0);
        c3x = svmla_lane_f64(c3x, bx, a2323, 1);
    }

    svst1_f64(npred, c, c0x);
    svst1_f64(npred_m_gt1, c + ldc, c1x);
    svst1_f64(npred_m_gt2, c + ldc * 2, c2x);
    svst1_f64(npred_m_gt3, c + ldc * 3, c3x);
}

void bl_dgemm_ukr_sve_8x4(int k,
                      int m,
                      int n,
                      const double *restrict a,
                      const double *restrict b,
                      double *restrict c,
                      unsigned long long ldc,
                      aux_t *data)
{
    register svfloat64_t ax;
    register svfloat64_t bx;
    register float64_t aval;

    #define REG_DEF(i) register svfloat64_t c ## i ## x;
    REPEAT_8(REG_DEF)
    #undef REG_DEF

    svbool_t npred = svwhilelt_b64_u64(0, n);
    #define PRED_DEF(i) svbool_t npred_m_gt ## i = svmov_z(npred, svdup_b64(m > i));
    REPEAT_1_8(PRED_DEF)
    #undef PRED_DEF

    #define LOAD_REG(i) c ## i ## x = svld1_f64(npred, c + ldc * i);
    REPEAT_8(LOAD_REG)
    #undef LOAD_REG

    for (int kk = 0; kk < k; kk++)
    {
        bx = svld1_f64(npred, b + kk * n);

        aval = *(a + kk * m);
        ax = svdup_f64(aval);
        c0x = svmla_f64_m(npred, c0x, bx, ax);

        #define BLOCK(i) \
            aval = *(a + kk * m + i); \
            ax = svdup_f64(aval); \
            c ## i ## x = svmla_f64_m(npred_m_gt ## i, c ## i ## x, bx, ax);
        REPEAT_1_8(BLOCK)
        #undef BLOCK
    }

    svst1_f64(npred, c, c0x);
    #define STORE_REG(i) svst1_f64(npred_m_gt ## i, c + ldc * i, c ## i ## x);
    REPEAT_1_8(STORE_REG)
    #undef STORE_REG
}

void bl_dgemm_ukr_sve_4x4_butterfly(int k,
                                    int m,
                                    int n,
                                    const double *restrict a,
                                    const double *restrict b,
                                    double *restrict c,
                                    unsigned long long ldc,
                                    aux_t *data)
{
    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c00_11_22_33, c30_01_12_23, c20_31_02_13, c10_21_32_03;

    register svbool_t mpred = svwhilelt_b64_u64(0, m);
    register svbool_t npred = svwhilelt_b64_u64(0, n);
    // the higest bit is set in this predicate [1, 0, 0, 0]
    register svbool_t rotate_pred = svnot_z(svptrue_b64(), svwhilelt_b64_u64(0, svcntd() - 1));

    // ### loading C: gather values into the C registers ###

    // [0, 1, 2, 3] * ldc * sizeof(double)
    register svuint64_t row_offset = svindex_u64(0, ldc * sizeof(double));
    // [0, 1, 2, 3] * sizeof(double)
    register svuint64_t column_offset = svindex_u64(0, sizeof(double));

    register svuint64_t offsets;
    register svbool_t load_store_pred;
    register uint64_t max_row_offset = ldc * sizeof(double) * m;

    offsets = svadd_u64_z(npred, row_offset, column_offset);
    // this masks off locations where B is invalid
    load_store_pred = svcmplt_n_u64(npred, row_offset, max_row_offset);
    c00_11_22_33 = svld1_gather_u64offset_f64(load_store_pred, c, offsets);

    // here we rotate the row vector: [0, 1, 2, 3] -> [3, 0, 1, 2]
    row_offset = svsplice_u64(rotate_pred, row_offset, row_offset);
    offsets = svadd_u64_z(npred, row_offset, column_offset);
    load_store_pred = svcmplt_n_u64(npred, row_offset, max_row_offset);
    c30_01_12_23 = svld1_gather_u64offset_f64(load_store_pred, c, offsets);

    // [3, 0, 1, 2] -> [2, 3, 0, 1]
    row_offset = svsplice_u64(rotate_pred, row_offset, row_offset);
    offsets = svadd_u64_z(npred, row_offset, column_offset);
    load_store_pred = svcmplt_n_u64(npred, row_offset, max_row_offset);
    c20_31_02_13 = svld1_gather_u64offset_f64(load_store_pred, c, offsets);

    // [2, 3, 0, 1] -> [1, 2, 3, 0]
    row_offset = svsplice_u64(rotate_pred, row_offset, row_offset);
    offsets = svadd_u64_z(npred, row_offset, column_offset);
    load_store_pred = svcmplt_n_u64(npred, row_offset, max_row_offset);
    c10_21_32_03 = svld1_gather_u64offset_f64(load_store_pred, c, offsets);

    // ### mult-add while rotating A ###
    for (int kk = 0; kk < k; kk++)
    {
        ax = svld1_f64(mpred, a + kk * m); 
        bx = svld1_f64(npred, b + kk * n);

        c00_11_22_33 = svmla_f64_m(npred, c00_11_22_33, bx, ax);

        // [0, 1, 2, 3] -> [3, 0, 1, 2]
        ax = svsplice_f64(rotate_pred, ax, ax);
        c30_01_12_23 = svmla_f64_m(npred, c30_01_12_23, bx, ax);

        // [3, 0, 1, 2] -> [2, 3, 0, 1]
        ax = svsplice_f64(rotate_pred, ax, ax);
        c20_31_02_13 = svmla_f64_m(npred, c20_31_02_13, bx, ax);

        // [2, 3, 0, 1] -> [1, 2, 3, 0]
        ax = svsplice_f64(rotate_pred, ax, ax);
        c10_21_32_03 = svmla_f64_m(npred, c10_21_32_03, bx, ax);
    }

    // ### storing C ###
    row_offset = svindex_u64(0, ldc * sizeof(double));

    offsets = svadd_u64_z(npred, row_offset, column_offset);
    load_store_pred = svcmplt_n_u64(npred, row_offset, max_row_offset);
    svst1_scatter_u64offset_f64(load_store_pred, c, offsets, c00_11_22_33);

    row_offset = svsplice_u64(rotate_pred, row_offset, row_offset);
    offsets = svadd_u64_z(npred, row_offset, column_offset);
    load_store_pred = svcmplt_n_u64(npred, row_offset, max_row_offset);
    svst1_scatter_u64offset_f64(load_store_pred, c, offsets, c30_01_12_23);

    row_offset = svsplice_u64(rotate_pred, row_offset, row_offset);
    offsets = svadd_u64_z(npred, row_offset, column_offset);
    load_store_pred = svcmplt_n_u64(npred, row_offset, max_row_offset);
    svst1_scatter_u64offset_f64(load_store_pred, c, offsets, c20_31_02_13);

    row_offset = svsplice_u64(rotate_pred, row_offset, row_offset);
    offsets = svadd_u64_z(npred, row_offset, column_offset);
    load_store_pred = svcmplt_n_u64(npred, row_offset, max_row_offset);
    svst1_scatter_u64offset_f64(load_store_pred, c, offsets, c10_21_32_03);
}

void bl_dgemm_ukr_sve_4x8(
    int k,
    int m,
    int n,
    const double *restrict a,
    const double *restrict b,
    double *restrict c,
    unsigned long long ldc,
    aux_t *data)
{
    register svfloat64_t ax;
    register svfloat64_t bx_low, bx_high;

    register float64_t aval;
    register svfloat64_t c0_low, c0_high, c1_low, c1_high, c2_low, c2_high, c3_low, c3_high;


    uint64_t len = svcntd();
    svbool_t npred_low = svwhilelt_b64_u64(0, n);
    svbool_t npred_high = svwhilelt_b64_u64(len, n);
    svbool_t npred_low_m_gt1 = svmov_z(npred_low, svdup_b64(m > 1));
    svbool_t npred_high_m_gt1 = svmov_z(npred_high, svdup_b64(m > 1));
    svbool_t npred_low_m_gt2= svmov_z(npred_low, svdup_b64(m > 2));
    svbool_t npred_high_m_gt2 = svmov_z(npred_high, svdup_b64(m > 2));
    svbool_t npred_low_m_gt3 = svmov_z(npred_low, svdup_b64(m > 3));
    svbool_t npred_high_m_gt3 = svmov_z(npred_high, svdup_b64(m > 3));

    c0_low = svld1_f64(npred_low, c);
    c0_high = svld1_f64(npred_low, c + len);
    c1_low = svld1_f64(npred_low_m_gt1, c + ldc);
    c1_high = svld1_f64(npred_high_m_gt1, c + ldc + len);
    c2_low = svld1_f64(npred_low_m_gt2, c + ldc * 2);
    c2_high = svld1_f64(npred_high_m_gt2, c + ldc * 2 + len);
    c3_low = svld1_f64(npred_low_m_gt3, c + ldc * 3);
    c3_high = svld1_f64(npred_high_m_gt3, c + ldc * 3 + len);

    for (int kk = 0; kk < k; kk++)
    {
        bx_low = svld1_f64(npred_low, b + kk * n);
        bx_high = svld1_f64(npred_high, b + kk * n + len);

        aval = *(a + kk * m);
        ax = svdup_f64(aval);
        c0_low = svmla_f64_m(npred_low, c0_low, bx_low, ax);
        c0_high = svmla_f64_m(npred_high, c0_high, bx_high, ax);

        aval = *(a + kk * m + 1);
        ax = svdup_f64(aval);
        c1_low = svmla_f64_m(npred_low_m_gt1, c1_low, bx_low, ax);
        c1_high = svmla_f64_m(npred_high_m_gt1, c1_high, bx_high, ax);

        aval = *(a + kk * m + 2);
        ax = svdup_f64(aval);
        c2_low = svmla_f64_m(npred_low_m_gt2, c2_low, bx_low, ax);
        c2_high = svmla_f64_m(npred_high_m_gt2, c2_high, bx_high, ax);

        aval = *(a + kk * m + 3);
        ax = svdup_f64(aval);
        c3_low = svmla_f64_m(npred_low_m_gt3, c3_low, bx_low, ax);
        c3_high = svmla_f64_m(npred_high_m_gt3, c3_high, bx_high, ax);
    }

    svst1_f64(npred_low, c, c0_low);
    svst1_f64(npred_high, c + len, c0_high);
    svst1_f64(npred_low_m_gt1, c + ldc, c1_low);
    svst1_f64(npred_high_m_gt1, c + ldc + len, c1_high);
    svst1_f64(npred_low_m_gt2, c + ldc * 2, c2_low);
    svst1_f64(npred_high_m_gt2, c + ldc * 2 + len, c2_high);
    svst1_f64(npred_low_m_gt3, c + ldc * 3, c3_low);
    svst1_f64(npred_high_m_gt3, c + ldc * 3 + len, c3_high);
}

void bl_dgemm_ukr_sve_4x12(
    int k,
    int m,
    int n,
    const double *restrict a,
    const double *restrict b,
    double *restrict c,
    unsigned long long ldc,
    aux_t *data)
{
    register svfloat64_t ax;
    register svfloat64_t bx_low, bx_mid, bx_high;

    register float64_t aval;

#define REG_DEF(i) register svfloat64_t c##i##_low, c##i##_mid, c##i##_high;
    REPEAT_4(REG_DEF)
#undef REG_DEF

    uint64_t len = svcntd();
    svbool_t npred0_low = svwhilelt_b64_u64(0, n);
    svbool_t npred0_mid = svwhilelt_b64_u64(len, n);
    svbool_t npred0_high = svwhilelt_b64_u64(len * 2, n);
#define PRED_DEF(i) \
    svbool_t npred##i##_low = svmov_z(npred0_low, svdup_b64(m > i)); \
    svbool_t npred##i##_mid = svmov_z(npred0_mid, svdup_b64(m > i)); \
    svbool_t npred##i##_high = svmov_z(npred0_high, svdup_b64(m > i)); 
    REPEAT_1_4(PRED_DEF)
#undef PRED_DEF

#define LOAD(i) \
    c##i##_low = svld1_f64(npred##i##_low, c + ldc * i); \
    c##i##_mid = svld1_f64(npred##i##_mid, c + ldc * i + len); \
    c##i##_high = svld1_f64(npred##i##_high, c + ldc * i + len * 2);
    REPEAT_4(LOAD)
#undef LOAD

    for (int kk = 0; kk < k; kk++)
    {
        bx_low = svld1_f64(npred0_low, b + kk * n);
        bx_mid = svld1_f64(npred0_mid, b + kk * n + len);
        bx_high = svld1_f64(npred0_high, b + kk * n + len * 2);

#define BLOCK(i) \
        aval = *(a + kk * m + i); \
        ax = svdup_f64(aval); \
        c##i##_low = svmla_f64_m(npred##i##_low, c##i##_low, bx_low, ax); \
        c##i##_mid = svmla_f64_m(npred##i##_mid, c##i##_mid, bx_mid, ax); \
        c##i##_high = svmla_f64_m(npred##i##_high, c##i##_high, bx_high, ax);
        REPEAT_4(BLOCK)
#undef BLOCK
    }

#define STORE(i) \
    svst1_f64(npred##i##_low, c + ldc * i, c##i##_low); \
    svst1_f64(npred##i##_mid, c + ldc * i + len, c##i##_mid); \
    svst1_f64(npred##i##_high, c + ldc * i + len * 2, c##i##_high); 
    REPEAT_4(STORE)
#undef STORE
}

// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//
