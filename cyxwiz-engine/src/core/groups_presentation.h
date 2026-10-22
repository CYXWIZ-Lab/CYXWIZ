#pragma once

// Words of Data Studio's Explore tabs Groups and Projection (TOFIX129 D5,
// approved mockup d5_clustering_mockup.html): the groups and how good they
// are, what each group is, how it compares with a column, how many groups to
// pick, what a projection keeps, and the CSV exports. Answers in, text out, so
// the tabs and their test agree. No ImGui, no queries.

#include "groups_explore.h"

#include <string>
#include <vector>

namespace cyxwiz::groups_view {

const char* MethodLabel(groups::Method m);
const char* ProjectionLabel(groups::ProjectionMethod m);
// Groups are shown from 1: label 0 -> "Group 1"; -1 -> "Noise".
std::string GroupName(int label);
// A share as a percent with one decimal: 0.6884 -> "68.8%".
std::string Share(double share);
// A score in three decimals: 0.4472 -> "0.447"; "" for NaN.
std::string Score(double v);

// The silhouette in words (Kaufman and Rousseeuw): "strong" (> 0.70),
// "reasonable" (> 0.50), "weak" (> 0.25), "no real" structure.
const char* SilhouetteWords(double silhouette);
// The adjusted Rand in words: "almost the same" (>= 0.9), "close" (0.65),
// "partly alike" (0.3), "hardly alike".
const char* AgreementWords(double ari);

// "3 groups of 87, 123 and 132 rows; silhouette 0.447: weak structure." /
// "2 groups and 33 rows of noise; ...".
std::string Summary(const groups::GroupsAnswer& a);
// What the method says about itself: "inertia 379.4 after 4 iterations",
// "cophenetic correlation 0.835: the tree keeps the rows' distances well",
// "BIC 2553.6, AIC 2480.2"; "" when nothing.
std::string MethodLine(const groups::GroupsAnswer& a);
// What a group is, from its means in standard deviations of the column:
// "high flipper_length_mm, high body_mass_g, low bill_depth_mm" (at most
// three, |z| >= 0.5, largest first); "close to the average" when none.
std::string Profile(const groups::GroupsAnswer& a, size_t group);
// "Group 2 is all Gentoo (123). Group 1 is mostly Chinstrap (63 of 87). ..."
// and the agreement: "Agreement with species (adjusted Rand) 0.793: close."
std::string CompareSentence(const groups::GroupsAnswer& a);
// "4 rows have a missing value and are left out." ("" when none).
std::string LeftOut(int64_t rows, int64_t used);

// How many groups, worded with its reason.
std::string HowManySentence(const groups::HowManyAnswer& a);

// Projection: what it shows. PCA: "Two components keep 88% of the variation."
// and each of the first two components in words; t-SNE / UMAP: the rows used.
std::string ProjectionSentence(const groups::ProjectionAnswer& a);
// A PCA component in words from its loadings (|l| >= 0.4): "flipper_length_mm,
// body_mass_g and bill_length_mm together, against bill_depth_mm".
std::string ComponentWords(const std::vector<double>& loadings, const std::vector<std::string>& columns);
// The axis titles: "component 1 (68.8%)", "t-SNE 1", "UMAP 2".
std::string AxisTitle(const groups::ProjectionAnswer& a, int axis);

// CSV: a row per group (rows, silhouette, the means), then a row per row used
// (its group, the scatter's coordinates, Mixture's probabilities).
std::string GroupsCsv(const groups::GroupsAnswer& a);
// A row per point: x, y, and the colour column.
std::string ProjectionCsv(const groups::ProjectionAnswer& a);

}  // namespace cyxwiz::groups_view
