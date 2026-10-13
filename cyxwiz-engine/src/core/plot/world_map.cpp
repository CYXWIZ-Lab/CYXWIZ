#include "world_map.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>

namespace cyxwiz::plot {

namespace {

struct RawCountry {
    const char* name;
    const char* name_long;
    const char* admin;
    const char* formal;
    const char* iso_a3;
    const char* iso_a2;
    const char* continent;
    int first_ring, ring_count;
};
struct RawRing {
    int first_point, points;
};

#include "world_countries_110m.inc"

// "Côte d'Ivoire" -> "cotedivoire": lower case letters and digits only
// (bytes above 127 are dropped, so accents do not matter either way).
std::string Key(const std::string& text) {
    std::string k;
    for (unsigned char ch : text)
        if (ch < 128 && std::isalnum(ch)) k += static_cast<char>(std::tolower(ch));
    return k;
}

// Names tables often use that Natural Earth does not.
const std::pair<const char*, const char*> kAliases[] = {
    {"UK", "GBR"}, {"Great Britain", "GBR"}, {"Britain", "GBR"}, {"England", "GBR"},
    {"US", "USA"}, {"America", "USA"}, {"United States", "USA"},
    {"Czech Republic", "CZE"}, {"Russian Federation", "RUS"}, {"Korea, Rep.", "KOR"}, {"Republic of Korea", "KOR"},
    {"Korea, Dem. People's Rep.", "PRK"}, {"Iran, Islamic Rep.", "IRN"}, {"Egypt, Arab Rep.", "EGY"},
    {"Venezuela, RB", "VEN"}, {"Syrian Arab Republic", "SYR"}, {"Lao PDR", "LAO"}, {"Viet Nam", "VNM"},
    {"Congo, Dem. Rep.", "COD"}, {"DR Congo", "COD"}, {"DRC", "COD"}, {"Congo, Rep.", "COG"},
    {"Ivory Coast", "CIV"}, {"Cote d'Ivoire", "CIV"}, {"Turkiye", "TUR"}, {"Burma", "MMR"},
    {"Slovak Republic", "SVK"}, {"Kyrgyz Republic", "KGZ"}, {"Yemen, Rep.", "YEM"}, {"Gambia, The", "GMB"},
    {"Bahamas, The", "BHS"}, {"Eswatini", "SWZ"}, {"Swaziland", "SWZ"}, {"East Timor", "TLS"},
    {"Macedonia", "MKD"}, {"North Macedonia", "MKD"}, {"Kosovo", "XK"}, {"XKX", "XK"},
};

struct Index {
    std::vector<Country> countries;
    std::unordered_map<std::string, int> by_key;
    std::vector<double> area;  // of the largest ring, to pick the smallest under a point
};

const Index& TheIndex() {
    static const Index index = [] {
        Index ix;
        for (const RawCountry& rc : kRawCountries) {
            Country c;
            c.name = rc.name;
            c.name_long = rc.name_long;
            c.iso_a3 = rc.iso_a3;
            c.iso_a2 = rc.iso_a2;
            c.continent = rc.continent;
            bool first = true;
            double area = 0;
            for (int r = rc.first_ring; r < rc.first_ring + rc.ring_count; ++r) {
                const RawRing& ring = kRawRings[r];
                std::vector<double> pts;
                pts.reserve(static_cast<size_t>(ring.points) * 2);
                double a = 0;
                for (int k = 0; k < ring.points; ++k) {
                    const double lon = kRawCoords[(ring.first_point + k) * 2] / 100.0;
                    const double lat = kRawCoords[(ring.first_point + k) * 2 + 1] / 100.0;
                    pts.push_back(lon);
                    pts.push_back(lat);
                    c.lon_min = first ? lon : std::min(c.lon_min, lon);
                    c.lon_max = first ? lon : std::max(c.lon_max, lon);
                    c.lat_min = first ? lat : std::min(c.lat_min, lat);
                    c.lat_max = first ? lat : std::max(c.lat_max, lat);
                    first = false;
                }
                for (size_t k = 0; k + 1 < pts.size(); k += 2) {
                    const size_t j = (k + 2) % pts.size();
                    a += pts[k] * pts[j + 1] - pts[j] * pts[k + 1];
                }
                area += std::fabs(a) / 2;
                c.rings.push_back(std::move(pts));
            }
            const int i = static_cast<int>(ix.countries.size());
            for (const char* n : {rc.name, rc.name_long, rc.admin, rc.formal, rc.iso_a3, rc.iso_a2})
                if (n && *n) ix.by_key.emplace(Key(n), i);
            ix.countries.push_back(std::move(c));
            ix.area.push_back(area);
        }
        for (const auto& [alias, code] : kAliases) {
            auto it = ix.by_key.find(Key(code));
            if (it != ix.by_key.end()) ix.by_key.emplace(Key(alias), it->second);
        }
        return ix;
    }();
    return index;
}

bool Inside(const std::vector<double>& ring, double x, double y) {
    bool in = false;
    const size_t n = ring.size() / 2;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const double xi = ring[i * 2], yi = ring[i * 2 + 1], xj = ring[j * 2], yj = ring[j * 2 + 1];
        if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) in = !in;
    }
    return in;
}

}  // namespace

const std::vector<Country>& WorldCountries() {
    return TheIndex().countries;
}

int FindCountry(const std::string& text) {
    const Index& ix = TheIndex();
    const std::string k = Key(text);
    if (k.empty()) return -1;
    auto it = ix.by_key.find(k);
    return it == ix.by_key.end() ? -1 : it->second;
}

int CountryAt(double lon, double lat) {
    const Index& ix = TheIndex();
    int best = -1;
    for (size_t i = 0; i < ix.countries.size(); ++i) {
        const Country& c = ix.countries[i];
        if (lon < c.lon_min || lon > c.lon_max || lat < c.lat_min || lat > c.lat_max) continue;
        for (const auto& ring : c.rings)
            if (Inside(ring, lon, lat)) {
                if (best < 0 || ix.area[i] < ix.area[static_cast<size_t>(best)]) best = static_cast<int>(i);
                break;
            }
    }
    return best;
}

}  // namespace cyxwiz::plot
