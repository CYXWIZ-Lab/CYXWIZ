#include "time_steps.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::series {

namespace {

constexpr double kDay = kDaySeconds;
constexpr double kMonth = 30.436875 * kDay;  // the mean Gregorian month

double Median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

std::string Trim(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

// ---- dates (proleptic Gregorian; Howard Hinnant's algorithms)

double CivilToSeconds(int y, int m, int d, int hour, int minute, int second) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const long long days = static_cast<long long>(era) * 146097 + static_cast<long long>(doe) - 719468;
    return static_cast<double>(days) * kDay + hour * 3600.0 + minute * 60.0 + second;
}

void SecondsToCivil(double seconds, int& y, int& m, int& d, int& hour, int& minute, int& second) {
    const double days_d = std::floor(seconds / kDay);
    double rest = seconds - days_d * kDay;
    long long z = static_cast<long long>(days_d) + 719468;
    const long long era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const long long year = static_cast<long long>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    m = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
    y = static_cast<int>(year + (m <= 2));
    hour = static_cast<int>(rest / 3600.0);
    rest -= hour * 3600.0;
    minute = static_cast<int>(rest / 60.0);
    second = static_cast<int>(std::lround(rest - minute * 60.0));
}

std::string DateLabel(double seconds, const Step& step) {
    int y, m, d, hh, mm, ss;
    SecondsToCivil(seconds, y, m, d, hh, mm, ss);
    char buf[40];
    if (step.seconds > 0.0 && step.seconds < kDay) std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d", y, m, d, hh, mm);
    else std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", y, m, d);
    return buf;
}

Step StepOf(const std::vector<double>& t, TimeKind kind) {
    Step s;
    if (kind == TimeKind::Order) {
        s.seconds = 1.0;
        s.even = true;
        return s;
    }
    if (t.size() < 2) return s;
    std::vector<double> gaps(t.size() - 1);
    for (size_t i = 1; i < t.size(); ++i) gaps[i - 1] = t[i] - t[i - 1];
    const double median = Median(gaps);
    if (!(median > 0.0)) {
        s.name = "no step";
        return s;
    }
    double lo = 0.9 * median, hi = 1.1 * median;
    if (kind == TimeKind::Numbers) {
        s.seconds = median;
        s.name = "every " + Trim(median);
    } else {
        const double days = median / kDay;
        if (days >= 27.5 && days <= 31.5) {
            s.name = "monthly";
            s.seconds = kMonth;
            s.natural_period = 12;
            lo = 28.0 * kDay;
            hi = 31.0 * kDay;
        } else if (days >= 89.0 && days <= 93.0) {
            s.name = "quarterly";
            s.seconds = 3.0 * kMonth;
            s.natural_period = 4;
            lo = 89.0 * kDay;
            hi = 93.0 * kDay;
        } else if (days >= 364.0 && days <= 367.0) {
            s.name = "yearly";
            s.seconds = 365.2425 * kDay;
            lo = 364.0 * kDay;
            hi = 367.0 * kDay;
        } else if (std::fabs(days - 7.0) < 0.01) {
            s.name = "weekly";
            s.seconds = 7.0 * kDay;
            s.natural_period = 52;
        } else if (std::fabs(days - 1.0) < 0.01) {
            s.name = "daily";
            s.seconds = kDay;
            s.natural_period = 7;
        } else if (std::fabs(median - 3600.0) < 1.0) {
            s.name = "hourly";
            s.seconds = 3600.0;
            s.natural_period = 24;
        } else if (std::fabs(median - 60.0) < 0.01) {
            s.name = "every minute";
            s.seconds = 60.0;
        } else {
            s.seconds = median;
            if (median >= kDay) s.name = "every " + Trim(days) + " days";
            else if (median >= 3600.0) s.name = "every " + Trim(median / 3600.0) + " hours";
            else if (median >= 60.0) s.name = "every " + Trim(median / 60.0) + " minutes";
            else s.name = "every " + Trim(median) + " seconds";
        }
    }
    size_t within = 0;
    for (const double g : gaps)
        if (g >= lo && g <= hi) ++within;
    s.even = static_cast<double>(within) >= 0.95 * static_cast<double>(gaps.size());
    return s;
}

std::vector<double> FutureTimes(const std::vector<double>& t, const Step& step, TimeKind kind, int count) {
    std::vector<double> out;
    if (t.empty() || count <= 0) return out;
    const double last = t.back();
    const int months = step.name == "monthly" ? 1 : step.name == "quarterly" ? 3 : step.name == "yearly" ? 12 : 0;
    if (kind == TimeKind::Dates && months > 0) {
        int y, m, d, hh, mm, ss;
        SecondsToCivil(last, y, m, d, hh, mm, ss);
        for (int k = 1; k <= count; ++k) {
            int total = (y * 12 + (m - 1)) + months * k;
            const int ny = total / 12, nm = total % 12 + 1;
            static const int kLength[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            int length = kLength[nm - 1];
            if (nm == 2 && ((ny % 4 == 0 && ny % 100 != 0) || ny % 400 == 0)) length = 29;
            out.push_back(CivilToSeconds(ny, nm, std::min(d, length), hh, mm, ss));
        }
        return out;
    }
    const double size = kind == TimeKind::Order ? 1.0 : step.seconds;
    for (int k = 1; k <= count; ++k) out.push_back(last + size * k);
    return out;
}

}  // namespace cyxwiz::series
