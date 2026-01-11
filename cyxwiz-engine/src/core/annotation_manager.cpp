#include "annotation_manager.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace cyxwiz {

// Static empty class list for returning when dataset not found
const std::vector<std::string> AnnotationManager::empty_class_list_;

// ============================================================================
// Enum String Conversions
// ============================================================================

std::string AnnotationTypeToString(AnnotationType type) {
    switch (type) {
        case AnnotationType::BBox:    return "bbox";
        case AnnotationType::Polygon: return "polygon";
        case AnnotationType::Mask:    return "mask";
        default:                      return "polygon";
    }
}

AnnotationType StringToAnnotationType(const std::string& str) {
    if (str == "bbox")    return AnnotationType::BBox;
    if (str == "polygon") return AnnotationType::Polygon;
    if (str == "mask")    return AnnotationType::Mask;
    return AnnotationType::Polygon;
}

// ============================================================================
// Annotation JSON Serialization
// ============================================================================

nlohmann::json Annotation::ToJson() const {
    nlohmann::json j;
    j["id"] = id;
    j["class_id"] = class_id;
    j["class_name"] = class_name;
    j["type"] = AnnotationTypeToString(type);
    j["area"] = area;

    // BBox
    j["bbox"] = {
        {"x", bbox.x},
        {"y", bbox.y},
        {"width", bbox.width},
        {"height", bbox.height},
        {"angle", bbox.angle}
    };

    // Polygon
    nlohmann::json poly_json = nlohmann::json::array();
    for (const auto& pt : polygon) {
        poly_json.push_back({pt.x, pt.y});
    }
    j["polygon"] = poly_json;

    // RLE
    j["rle"] = rle;

    return j;
}

Annotation Annotation::FromJson(const nlohmann::json& j) {
    Annotation ann;

    if (j.contains("id")) ann.id = j["id"];
    if (j.contains("class_id")) ann.class_id = j["class_id"];
    if (j.contains("class_name")) ann.class_name = j["class_name"];
    if (j.contains("type")) ann.type = StringToAnnotationType(j["type"]);
    if (j.contains("area")) ann.area = j["area"];

    // BBox
    if (j.contains("bbox")) {
        const auto& b = j["bbox"];
        if (b.contains("x")) ann.bbox.x = b["x"];
        if (b.contains("y")) ann.bbox.y = b["y"];
        if (b.contains("width")) ann.bbox.width = b["width"];
        if (b.contains("height")) ann.bbox.height = b["height"];
        if (b.contains("angle")) ann.bbox.angle = b["angle"];
    }

    // Polygon
    if (j.contains("polygon")) {
        for (const auto& pt : j["polygon"]) {
            if (pt.size() >= 2) {
                ann.polygon.push_back({pt[0].get<float>(), pt[1].get<float>()});
            }
        }
    }

    // RLE
    if (j.contains("rle")) {
        ann.rle = j["rle"].get<std::vector<int>>();
    }

    return ann;
}

// ============================================================================
// ImageAnnotations Implementation
// ============================================================================

int ImageAnnotations::AddAnnotation(const Annotation& ann) {
    Annotation new_ann = ann;
    new_ann.id = next_annotation_id++;
    annotations.push_back(new_ann);
    return new_ann.id;
}

bool ImageAnnotations::RemoveAnnotation(int ann_id) {
    auto it = std::find_if(annotations.begin(), annotations.end(),
                           [ann_id](const Annotation& a) { return a.id == ann_id; });
    if (it != annotations.end()) {
        annotations.erase(it);
        return true;
    }
    return false;
}

Annotation* ImageAnnotations::FindAnnotation(int ann_id) {
    auto it = std::find_if(annotations.begin(), annotations.end(),
                           [ann_id](const Annotation& a) { return a.id == ann_id; });
    return it != annotations.end() ? &(*it) : nullptr;
}

const Annotation* ImageAnnotations::FindAnnotation(int ann_id) const {
    auto it = std::find_if(annotations.begin(), annotations.end(),
                           [ann_id](const Annotation& a) { return a.id == ann_id; });
    return it != annotations.end() ? &(*it) : nullptr;
}

nlohmann::json ImageAnnotations::ToJson() const {
    nlohmann::json j;
    j["image_idx"] = image_idx;
    j["width"] = width;
    j["height"] = height;
    j["next_annotation_id"] = next_annotation_id;

    nlohmann::json anns = nlohmann::json::array();
    for (const auto& ann : annotations) {
        anns.push_back(ann.ToJson());
    }
    j["annotations"] = anns;

    return j;
}

ImageAnnotations ImageAnnotations::FromJson(const nlohmann::json& j) {
    ImageAnnotations img;

    if (j.contains("image_idx")) img.image_idx = j["image_idx"];
    if (j.contains("width")) img.width = j["width"];
    if (j.contains("height")) img.height = j["height"];
    if (j.contains("next_annotation_id")) img.next_annotation_id = j["next_annotation_id"];

    if (j.contains("annotations")) {
        for (const auto& ann_json : j["annotations"]) {
            img.annotations.push_back(Annotation::FromJson(ann_json));
        }
    }

    return img;
}

// ============================================================================
// AnnotationSet Implementation
// ============================================================================

int AnnotationSet::AddClass(const std::string& name) {
    // Check if already exists
    auto it = std::find(class_names.begin(), class_names.end(), name);
    if (it != class_names.end()) {
        return static_cast<int>(std::distance(class_names.begin(), it));
    }
    // Add new class
    class_names.push_back(name);
    return static_cast<int>(class_names.size() - 1);
}

int AnnotationSet::GetClassId(const std::string& name) const {
    auto it = std::find(class_names.begin(), class_names.end(), name);
    if (it != class_names.end()) {
        return static_cast<int>(std::distance(class_names.begin(), it));
    }
    return -1;
}

const std::string& AnnotationSet::GetClassName(int class_id) const {
    static const std::string empty;
    if (class_id >= 0 && class_id < static_cast<int>(class_names.size())) {
        return class_names[class_id];
    }
    return empty;
}

ImageAnnotations& AnnotationSet::GetOrCreateImage(size_t image_idx, int width, int height) {
    auto it = images.find(image_idx);
    if (it == images.end()) {
        ImageAnnotations img;
        img.image_idx = image_idx;
        img.width = width;
        img.height = height;
        images[image_idx] = img;
    }
    return images[image_idx];
}

ImageAnnotations* AnnotationSet::GetImage(size_t image_idx) {
    auto it = images.find(image_idx);
    return it != images.end() ? &(it->second) : nullptr;
}

const ImageAnnotations* AnnotationSet::GetImage(size_t image_idx) const {
    auto it = images.find(image_idx);
    return it != images.end() ? &(it->second) : nullptr;
}

size_t AnnotationSet::GetTotalAnnotationCount() const {
    size_t count = 0;
    for (const auto& [idx, img] : images) {
        count += img.annotations.size();
    }
    return count;
}

size_t AnnotationSet::GetAnnotatedImageCount() const {
    size_t count = 0;
    for (const auto& [idx, img] : images) {
        if (!img.annotations.empty()) {
            count++;
        }
    }
    return count;
}

nlohmann::json AnnotationSet::ToJson() const {
    nlohmann::json j;
    j["dataset_id"] = dataset_id;
    j["class_names"] = class_names;

    nlohmann::json imgs = nlohmann::json::object();
    for (const auto& [idx, img] : images) {
        imgs[std::to_string(idx)] = img.ToJson();
    }
    j["images"] = imgs;

    return j;
}

AnnotationSet AnnotationSet::FromJson(const nlohmann::json& j) {
    AnnotationSet set;

    if (j.contains("dataset_id")) set.dataset_id = j["dataset_id"];
    if (j.contains("class_names")) {
        set.class_names = j["class_names"].get<std::vector<std::string>>();
    }

    if (j.contains("images")) {
        for (auto& [key, val] : j["images"].items()) {
            size_t idx = std::stoull(key);
            set.images[idx] = ImageAnnotations::FromJson(val);
        }
    }

    return set;
}

// ============================================================================
// AnnotationManager Implementation
// ============================================================================

void AnnotationManager::CreateAnnotationSet(const std::string& dataset_id) {
    if (annotation_sets_.find(dataset_id) == annotation_sets_.end()) {
        AnnotationSet set;
        set.dataset_id = dataset_id;
        annotation_sets_[dataset_id] = set;
        spdlog::debug("AnnotationManager: Created annotation set for dataset '{}'", dataset_id);
    }
}

AnnotationSet* AnnotationManager::GetAnnotationSet(const std::string& dataset_id) {
    auto it = annotation_sets_.find(dataset_id);
    return it != annotation_sets_.end() ? &(it->second) : nullptr;
}

const AnnotationSet* AnnotationManager::GetAnnotationSet(const std::string& dataset_id) const {
    auto it = annotation_sets_.find(dataset_id);
    return it != annotation_sets_.end() ? &(it->second) : nullptr;
}

bool AnnotationManager::RemoveAnnotationSet(const std::string& dataset_id) {
    auto it = annotation_sets_.find(dataset_id);
    if (it != annotation_sets_.end()) {
        annotation_sets_.erase(it);
        spdlog::debug("AnnotationManager: Removed annotation set for dataset '{}'", dataset_id);
        return true;
    }
    return false;
}

bool AnnotationManager::HasAnnotationSet(const std::string& dataset_id) const {
    return annotation_sets_.find(dataset_id) != annotation_sets_.end();
}

int AnnotationManager::AddClass(const std::string& dataset_id, const std::string& class_name) {
    CreateAnnotationSet(dataset_id);
    return annotation_sets_[dataset_id].AddClass(class_name);
}

const std::vector<std::string>& AnnotationManager::GetClasses(const std::string& dataset_id) const {
    auto it = annotation_sets_.find(dataset_id);
    if (it != annotation_sets_.end()) {
        return it->second.class_names;
    }
    return empty_class_list_;
}

int AnnotationManager::AddAnnotation(const std::string& dataset_id, size_t image_idx,
                                      int class_id, AnnotationType type,
                                      const std::vector<Point2D>& polygon,
                                      const BoundingBox& bbox,
                                      int image_width, int image_height) {
    CreateAnnotationSet(dataset_id);
    auto& set = annotation_sets_[dataset_id];

    // Get or create image annotations
    auto& img = set.GetOrCreateImage(image_idx, image_width, image_height);

    // Create annotation
    Annotation ann;
    ann.class_id = class_id;
    ann.class_name = set.GetClassName(class_id);
    ann.type = type;

    if (type == AnnotationType::BBox) {
        ann.bbox = bbox;
        ann.area = bbox.width * bbox.height;
    } else if (type == AnnotationType::Polygon && !polygon.empty()) {
        ann.polygon = polygon;
        // Compute bbox from polygon
        ann.bbox = LabelingUtils::ContourToBBox(polygon, false);
        // Compute area using OpenCV
        ann.area = static_cast<float>(ShapeUtils::ContourArea(polygon));
        // Generate RLE if we have dimensions
        if (image_width > 0 && image_height > 0) {
            ann.rle = LabelingUtils::PolygonToRLE(polygon, image_width, image_height);
        }
    }

    int ann_id = img.AddAnnotation(ann);
    spdlog::debug("AnnotationManager: Added annotation {} to dataset '{}' image {}",
                  ann_id, dataset_id, image_idx);
    return ann_id;
}

bool AnnotationManager::RemoveAnnotation(const std::string& dataset_id, size_t image_idx, int ann_id) {
    auto* set = GetAnnotationSet(dataset_id);
    if (!set) return false;

    auto* img = set->GetImage(image_idx);
    if (!img) return false;

    return img->RemoveAnnotation(ann_id);
}

std::vector<Annotation>* AnnotationManager::GetAnnotations(const std::string& dataset_id, size_t image_idx) {
    auto* set = GetAnnotationSet(dataset_id);
    if (!set) return nullptr;

    auto* img = set->GetImage(image_idx);
    return img ? &(img->annotations) : nullptr;
}

const std::vector<Annotation>* AnnotationManager::GetAnnotations(const std::string& dataset_id, size_t image_idx) const {
    const auto* set = GetAnnotationSet(dataset_id);
    if (!set) return nullptr;

    const auto* img = set->GetImage(image_idx);
    return img ? &(img->annotations) : nullptr;
}

void AnnotationManager::ClearImageAnnotations(const std::string& dataset_id, size_t image_idx) {
    auto* set = GetAnnotationSet(dataset_id);
    if (!set) return;

    auto* img = set->GetImage(image_idx);
    if (img) {
        img->annotations.clear();
    }
}

// ============================================================================
// Export Functions
// ============================================================================

bool AnnotationManager::ExportCOCO(const std::string& dataset_id, const std::string& output_path,
                                    const std::string& image_dir) const {
    const auto* set = GetAnnotationSet(dataset_id);
    if (!set) {
        spdlog::error("AnnotationManager: Dataset '{}' not found for COCO export", dataset_id);
        return false;
    }

    try {
        nlohmann::json coco;

        // Info section
        auto now = std::time(nullptr);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S");

        coco["info"] = {
            {"description", "CyxWiz Annotation Export"},
            {"version", "1.0"},
            {"year", 2026},
            {"date_created", ss.str()}
        };

        // Categories
        nlohmann::json categories = nlohmann::json::array();
        for (size_t i = 0; i < set->class_names.size(); ++i) {
            categories.push_back({
                {"id", static_cast<int>(i + 1)},  // COCO uses 1-indexed
                {"name", set->class_names[i]},
                {"supercategory", "object"}
            });
        }
        coco["categories"] = categories;

        // Images and annotations
        nlohmann::json images_json = nlohmann::json::array();
        nlohmann::json annotations_json = nlohmann::json::array();
        int global_ann_id = 1;

        for (const auto& [img_idx, img] : set->images) {
            // Image entry
            int image_id = static_cast<int>(img_idx + 1);  // 1-indexed
            std::string filename = "image_" + std::to_string(img_idx) + ".png";
            if (!image_dir.empty()) {
                filename = std::filesystem::path(image_dir).filename().string() + "/" + filename;
            }

            images_json.push_back({
                {"id", image_id},
                {"file_name", filename},
                {"width", img.width},
                {"height", img.height}
            });

            // Annotations for this image
            for (const auto& ann : img.annotations) {
                nlohmann::json ann_json;
                ann_json["id"] = global_ann_id++;
                ann_json["image_id"] = image_id;
                ann_json["category_id"] = ann.class_id + 1;  // 1-indexed
                ann_json["iscrowd"] = 0;
                ann_json["area"] = ann.area;

                // Bounding box [x, y, width, height]
                ann_json["bbox"] = {ann.bbox.x, ann.bbox.y, ann.bbox.width, ann.bbox.height};

                // Segmentation
                if (ann.type == AnnotationType::Polygon && !ann.polygon.empty()) {
                    // Polygon format: [[x1, y1, x2, y2, ...]]
                    nlohmann::json seg = nlohmann::json::array();
                    nlohmann::json poly = nlohmann::json::array();
                    for (const auto& pt : ann.polygon) {
                        poly.push_back(pt.x);
                        poly.push_back(pt.y);
                    }
                    seg.push_back(poly);
                    ann_json["segmentation"] = seg;
                } else if (ann.type == AnnotationType::Mask && !ann.rle.empty()) {
                    // RLE format
                    ann_json["segmentation"] = {
                        {"counts", ann.rle},
                        {"size", {img.height, img.width}}
                    };
                } else {
                    // Empty segmentation for bbox-only
                    ann_json["segmentation"] = nlohmann::json::array();
                }

                annotations_json.push_back(ann_json);
            }
        }

        coco["images"] = images_json;
        coco["annotations"] = annotations_json;

        // Write to file
        std::ofstream file(output_path);
        if (!file.is_open()) {
            spdlog::error("AnnotationManager: Failed to open '{}' for writing", output_path);
            return false;
        }

        file << coco.dump(2);
        file.close();

        spdlog::info("AnnotationManager: Exported COCO annotations to '{}'", output_path);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("AnnotationManager: COCO export failed: {}", e.what());
        return false;
    }
}

bool AnnotationManager::ExportYOLO(const std::string& dataset_id, const std::string& output_dir,
                                    const std::string& classes_file) const {
    const auto* set = GetAnnotationSet(dataset_id);
    if (!set) {
        spdlog::error("AnnotationManager: Dataset '{}' not found for YOLO export", dataset_id);
        return false;
    }

    try {
        // Create output directory if needed
        std::filesystem::create_directories(output_dir);

        // Write classes.txt if requested
        if (!classes_file.empty()) {
            std::ofstream cls_file(classes_file);
            if (cls_file.is_open()) {
                for (const auto& name : set->class_names) {
                    cls_file << name << "\n";
                }
                cls_file.close();
                spdlog::debug("AnnotationManager: Wrote classes to '{}'", classes_file);
            }
        }

        // Write one .txt per image
        for (const auto& [img_idx, img] : set->images) {
            if (img.annotations.empty()) continue;
            if (img.width <= 0 || img.height <= 0) continue;

            std::string filename = (std::filesystem::path(output_dir) /
                                   ("image_" + std::to_string(img_idx) + ".txt")).string();

            std::ofstream file(filename);
            if (!file.is_open()) {
                spdlog::warn("AnnotationManager: Failed to write '{}'", filename);
                continue;
            }

            for (const auto& ann : img.annotations) {
                // YOLO format: class_id center_x center_y width height (normalized)
                float cx = (ann.bbox.x + ann.bbox.width / 2) / img.width;
                float cy = (ann.bbox.y + ann.bbox.height / 2) / img.height;
                float w = ann.bbox.width / img.width;
                float h = ann.bbox.height / img.height;

                // Clamp to [0, 1]
                cx = std::max(0.0f, std::min(1.0f, cx));
                cy = std::max(0.0f, std::min(1.0f, cy));
                w = std::max(0.0f, std::min(1.0f, w));
                h = std::max(0.0f, std::min(1.0f, h));

                file << ann.class_id << " "
                     << std::fixed << std::setprecision(6)
                     << cx << " " << cy << " " << w << " " << h << "\n";
            }

            file.close();
        }

        spdlog::info("AnnotationManager: Exported YOLO annotations to '{}'", output_dir);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("AnnotationManager: YOLO export failed: {}", e.what());
        return false;
    }
}

bool AnnotationManager::ExportVOC(const std::string& dataset_id, const std::string& output_dir,
                                   const std::string& image_dir) const {
    const auto* set = GetAnnotationSet(dataset_id);
    if (!set) {
        spdlog::error("AnnotationManager: Dataset '{}' not found for VOC export", dataset_id);
        return false;
    }

    try {
        // Create output directory if needed
        std::filesystem::create_directories(output_dir);

        // Write one .xml per image
        for (const auto& [img_idx, img] : set->images) {
            if (img.annotations.empty()) continue;

            std::string filename = (std::filesystem::path(output_dir) /
                                   ("image_" + std::to_string(img_idx) + ".xml")).string();

            std::ofstream file(filename);
            if (!file.is_open()) {
                spdlog::warn("AnnotationManager: Failed to write '{}'", filename);
                continue;
            }

            std::string image_filename = "image_" + std::to_string(img_idx) + ".png";
            std::string image_path = image_dir.empty() ? image_filename :
                                     (std::filesystem::path(image_dir) / image_filename).string();

            // Write Pascal VOC XML
            file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
            file << "<annotation>\n";
            file << "  <folder>" << (image_dir.empty() ? "images" : image_dir) << "</folder>\n";
            file << "  <filename>" << image_filename << "</filename>\n";
            file << "  <path>" << image_path << "</path>\n";
            file << "  <source>\n";
            file << "    <database>CyxWiz</database>\n";
            file << "  </source>\n";
            file << "  <size>\n";
            file << "    <width>" << img.width << "</width>\n";
            file << "    <height>" << img.height << "</height>\n";
            file << "    <depth>3</depth>\n";
            file << "  </size>\n";
            file << "  <segmented>0</segmented>\n";

            for (const auto& ann : img.annotations) {
                file << "  <object>\n";
                file << "    <name>" << ann.class_name << "</name>\n";
                file << "    <pose>Unspecified</pose>\n";
                file << "    <truncated>0</truncated>\n";
                file << "    <difficult>0</difficult>\n";
                file << "    <bndbox>\n";
                file << "      <xmin>" << static_cast<int>(ann.bbox.x) << "</xmin>\n";
                file << "      <ymin>" << static_cast<int>(ann.bbox.y) << "</ymin>\n";
                file << "      <xmax>" << static_cast<int>(ann.bbox.x + ann.bbox.width) << "</xmax>\n";
                file << "      <ymax>" << static_cast<int>(ann.bbox.y + ann.bbox.height) << "</ymax>\n";
                file << "    </bndbox>\n";
                file << "  </object>\n";
            }

            file << "</annotation>\n";
            file.close();
        }

        spdlog::info("AnnotationManager: Exported VOC annotations to '{}'", output_dir);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("AnnotationManager: VOC export failed: {}", e.what());
        return false;
    }
}

// ============================================================================
// Training Integration
// ============================================================================

std::vector<float> AnnotationManager::GetSegmentationMask(const std::string& dataset_id,
                                                           size_t image_idx,
                                                           int width, int height) const {
    std::vector<float> mask(static_cast<size_t>(width) * height, 0.0f);

    const auto* anns = GetAnnotations(dataset_id, image_idx);
    if (!anns || anns->empty()) {
        return mask;
    }

    // Render each annotation with its class_id + 1 (0 = background)
    for (const auto& ann : *anns) {
        float class_value = static_cast<float>(ann.class_id + 1);

        if (ann.type == AnnotationType::Polygon && !ann.polygon.empty()) {
            // Create mask from polygon and overlay
            auto ann_mask = LabelingUtils::CreateMaskFromContour(ann.polygon, width, height, true);
            for (size_t i = 0; i < mask.size() && i < ann_mask.size(); ++i) {
                if (ann_mask[i] > 0.5f) {
                    mask[i] = class_value;
                }
            }
        } else if (ann.type == AnnotationType::Mask && !ann.rle.empty()) {
            // Decode RLE and overlay
            auto ann_mask = LabelingUtils::RLEToMask(ann.rle, width, height);
            for (size_t i = 0; i < mask.size() && i < ann_mask.size(); ++i) {
                if (ann_mask[i] > 0.5f) {
                    mask[i] = class_value;
                }
            }
        } else if (ann.type == AnnotationType::BBox) {
            // Fill bbox region
            int x1 = static_cast<int>(std::max(0.0f, ann.bbox.x));
            int y1 = static_cast<int>(std::max(0.0f, ann.bbox.y));
            int x2 = static_cast<int>(std::min(static_cast<float>(width), ann.bbox.x + ann.bbox.width));
            int y2 = static_cast<int>(std::min(static_cast<float>(height), ann.bbox.y + ann.bbox.height));

            for (int y = y1; y < y2; ++y) {
                for (int x = x1; x < x2; ++x) {
                    mask[y * width + x] = class_value;
                }
            }
        }
    }

    return mask;
}

std::vector<float> AnnotationManager::GetClassMask(const std::string& dataset_id,
                                                    size_t image_idx, int class_id,
                                                    int width, int height) const {
    std::vector<float> mask(static_cast<size_t>(width) * height, 0.0f);

    const auto* anns = GetAnnotations(dataset_id, image_idx);
    if (!anns || anns->empty()) {
        return mask;
    }

    for (const auto& ann : *anns) {
        if (ann.class_id != class_id) continue;

        if (ann.type == AnnotationType::Polygon && !ann.polygon.empty()) {
            auto ann_mask = LabelingUtils::CreateMaskFromContour(ann.polygon, width, height, true);
            for (size_t i = 0; i < mask.size() && i < ann_mask.size(); ++i) {
                if (ann_mask[i] > 0.5f) {
                    mask[i] = 1.0f;
                }
            }
        } else if (ann.type == AnnotationType::Mask && !ann.rle.empty()) {
            auto ann_mask = LabelingUtils::RLEToMask(ann.rle, width, height);
            for (size_t i = 0; i < mask.size() && i < ann_mask.size(); ++i) {
                if (ann_mask[i] > 0.5f) {
                    mask[i] = 1.0f;
                }
            }
        } else if (ann.type == AnnotationType::BBox) {
            int x1 = static_cast<int>(std::max(0.0f, ann.bbox.x));
            int y1 = static_cast<int>(std::max(0.0f, ann.bbox.y));
            int x2 = static_cast<int>(std::min(static_cast<float>(width), ann.bbox.x + ann.bbox.width));
            int y2 = static_cast<int>(std::min(static_cast<float>(height), ann.bbox.y + ann.bbox.height));

            for (int y = y1; y < y2; ++y) {
                for (int x = x1; x < x2; ++x) {
                    mask[y * width + x] = 1.0f;
                }
            }
        }
    }

    return mask;
}

// ============================================================================
// Persistence
// ============================================================================

nlohmann::json AnnotationManager::ToJson() const {
    nlohmann::json j = nlohmann::json::object();

    for (const auto& [id, set] : annotation_sets_) {
        j[id] = set.ToJson();
    }

    return j;
}

void AnnotationManager::FromJson(const nlohmann::json& j) {
    annotation_sets_.clear();

    for (auto& [key, val] : j.items()) {
        annotation_sets_[key] = AnnotationSet::FromJson(val);
    }

    spdlog::info("AnnotationManager: Loaded {} annotation sets", annotation_sets_.size());
}

void AnnotationManager::Clear() {
    annotation_sets_.clear();
    spdlog::debug("AnnotationManager: Cleared all annotations");
}

size_t AnnotationManager::GetTotalAnnotationCount() const {
    size_t count = 0;
    for (const auto& [id, set] : annotation_sets_) {
        count += set.GetTotalAnnotationCount();
    }
    return count;
}

std::vector<std::string> AnnotationManager::GetDatasetIds() const {
    std::vector<std::string> ids;
    for (const auto& [id, set] : annotation_sets_) {
        ids.push_back(id);
    }
    return ids;
}

} // namespace cyxwiz
