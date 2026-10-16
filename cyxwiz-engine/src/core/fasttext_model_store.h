#pragma once

// The latest trained fastText model of each saved vectors file, kept for the
// session (TOFIX152 P6): the saved .vec holds the vocabulary's rows only, so
// the Word Embeddings dialog builds the vectors of words the training never
// saw from the model's n-gram rows kept here. One model per file; a new run
// replaces it.

#include <cyxwiz/word_embeddings.h>

#include <memory>
#include <string>

namespace cyxwiz {

void KeepFastTextModel(const std::string& vectors_path, std::shared_ptr<const FastTextModel> model);
// Null when no run of this session trained the file.
std::shared_ptr<const FastTextModel> FindFastTextModel(const std::string& vectors_path);

}  // namespace cyxwiz
