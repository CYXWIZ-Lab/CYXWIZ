#pragma once

// Time steps of a series (TOFIX129 D6): what kind of time a column holds,
// the step between rows read from the dates (monthly, daily, ...), the
// dates that follow the last one, and civil dates to and from seconds since
// 1970. Pure functions shared by Data Studio's Series and Forecast tabs
// (series_explore) and the time series nodes (time_series_analysis_operators).

#include <string>
#include <vector>

namespace cyxwiz::series {

enum class TimeKind { Order, Numbers, Dates };

// A time step read from the dates (or numbers).
struct Step {
    std::string name;        // "monthly", "daily", "every 3 days", "every 0.5" (numbers), "" (order)
    double seconds = 0.0;    // dates: the step in seconds (a month: 30.436875 days); numbers: the step
    bool even = false;       // at least 95% of the gaps are the step (a month: 28 .. 31 days)
    int natural_period = 0;  // the period the step suggests (monthly 12, quarterly 4, daily 7, hourly 24), else 0
};

// The step of sorted times (dates in seconds, or numbers): its name, size,
// whether the gaps are even, and the period it suggests.
Step StepOf(const std::vector<double>& t, TimeKind kind);
// Dates `count` steps after the last one (calendar months for a monthly step).
std::vector<double> FutureTimes(const std::vector<double>& t, const Step& step, TimeKind kind, int count);
// A date in seconds since 1970 as "YYYY-MM-DD" (with " HH:MM" when the step is below a day).
std::string DateLabel(double seconds, const Step& step);
// seconds since 1970 of a civil date, and back.
double CivilToSeconds(int year, int month, int day, int hour = 0, int minute = 0, int second = 0);
void SecondsToCivil(double seconds, int& year, int& month, int& day, int& hour, int& minute, int& second);
// A number in a few digits ("0.5", "3").
std::string Trim(double v);

constexpr double kDaySeconds = 86400.0;

}  // namespace cyxwiz::series
