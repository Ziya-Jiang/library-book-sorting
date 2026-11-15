#pragma once
#include <cmath>
#include <stdexcept>
namespace seldon {
struct StartupDelta { int laps; double remainder; bool matches; };
inline StartupDelta startup_delta(double current, double saved, double tolerance = 0.3) {
    if (!std::isfinite(current) || !std::isfinite(saved))
        throw std::invalid_argument("Non-finite encoder reference");
    constexpr double tau = 6.283185307179586;
    const double delta = current - saved;
    if (std::abs(delta) > tau * 1000) throw std::invalid_argument("Invalid encoder turn count");
    const int laps = static_cast<int>(std::round(delta / tau));
    const double remainder = delta - laps * tau;
    return {laps, remainder, std::abs(remainder) <= tolerance};
}
}
