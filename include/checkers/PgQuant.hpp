#pragma once
// Host-side per-group step quantization for the RMS-relative uniform quant
// codec.  Shared verbatim by the compressor and the self-test so their
// numbers are guaranteed identical.
//
// Mirrors SOTA/EQ/compress_master.py::_compress_pergroup :
//   D_g   = 2 * eps * RMS_g
//   dcode = floor( (log(D_g) - lo) / step_q )          [FLOOR => D_hat <= D_g]
//   D_hat = exp(lo + dcode * step_q)
//   D_use = D_hat * shrink                             [shrink absorbs float ULP]
// The decoder recomputes D_use the same way, so encode/decode steps match.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace checkers {

struct PgSteps {
    std::vector<uint8_t> dcodes;   // [n_groups] u8   log-domain step codes
    std::vector<float>   shrink;   // [n_groups] f32  per-group ULP shrink
    std::vector<float>   d_use;    // [n_groups] f32  decoder-exact step (upload to GPU)
    double               pg_lo   = 0.0;
    double               pg_step = 0.0;
};

inline PgSteps quantize_steps(const std::vector<float>& grms, double eps, int scale_bits = 8) {
    const size_t ng = grms.size();
    PgSteps out;
    out.dcodes.resize(ng);
    out.shrink.resize(ng);
    out.d_use.resize(ng);
    if (ng == 0) return out;

    float amax = 0.0f;
    for (float r : grms) amax = std::max(amax, r);
    const double floor_r = std::max<double>(static_cast<double>(amax) * 1e-12,
                                            static_cast<double>(std::numeric_limits<float>::min()));

    std::vector<double> lD(ng);
    double lo =  std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < ng; ++i) {
        const double r  = std::max(static_cast<double>(grms[i]), floor_r);
        const double Dg = 2.0 * eps * r;
        lD[i] = std::log(Dg);
        lo = std::min(lo, lD[i]);
        hi = std::max(hi, lD[i]);
    }

    const double levels = static_cast<double>((1 << scale_bits) - 1);
    const double step_q = std::max(hi - lo, 1e-30) / levels;
    out.pg_lo   = lo;
    out.pg_step = step_q;

    const float shrink = 1.0f - 1e-6f;
    for (size_t i = 0; i < ng; ++i) {
        double code = std::floor((lD[i] - lo) / step_q);
        code = std::min(std::max(code, 0.0), levels);
        out.dcodes[i] = static_cast<uint8_t>(code);
        const double D_hat = std::exp(lo + code * step_q);   // <= D_g by floor
        out.shrink[i] = shrink;
        out.d_use[i]  = static_cast<float>(D_hat * static_cast<double>(shrink));
    }
    return out;
}

} // namespace checkers
