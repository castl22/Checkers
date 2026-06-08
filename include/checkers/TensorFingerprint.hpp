#pragma once

namespace checkers {

struct Fingerprint {
    double values[5];
};

struct FingerprintWeights {
    double w_mean     = 0.2;
    double w_variance = 0.2;
    double w_skewness = 0.2;
    double w_kurtosis = 0.2;
    double w_range    = 0.2;
};

}