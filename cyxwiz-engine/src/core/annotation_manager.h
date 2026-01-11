#pragma once

#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <nlohmann/json.hpp>
#include "utils/labeling_utils.h"
#include "utils/shape_utils.h"

namespace cyxwiz {

/**
 * Type of annotation
 */
enum class AnnotationType {
    BBox,       // Axis-aligned bounding box
    Polygon,    // Polygon mask (stored as polygon, can export as RLE)
    Mask        // Raw RLE mask
};

/**
 * Single annotation on an image
 */
struct Annotation {
    int id = 0;                     // Unique within image
    int class_id = 0;               // Index into class list
    std::string class_name;         // Cached class name
    AnnotationType type = AnnotationType::Polygon;

    // For BBox type - stores x, y, width, height
    BoundingBox bbox;

    // For Polygon/Mask type - polygon points
    std::vector<Point2D> polygon;

    // For Mask type - RLE encoded (COCO format counts)
    std::vector<int> rle;

    // Computed from polygon/mask
    float area = 0.0f;

    // JSON serialization
    nlohmann::json ToJson() const;
    static Annotation FromJson(const nlohmann::json& j);
};

/**
 * All annotations for one image in a dataset
 */
struct ImageAnnotations {
    size_t image_idx = 0;
    int width = 0;
    int height = 0;
    std::vector<Annotation> annotations;
    int next_annotation_id = 1;

    // Add annotation, returns assigned ID
    int AddAnnotation(const Annotation& ann);

    // Remove annotation by ID, returns true if found
    bool RemoveAnnotation(int ann_id);

    // Find annotation by ID
    Annotation* FindAnnotation(int ann_id);
    const Annotation* FindAnnotation(int ann_id) const;

    // JSON serialization
    nlohmann::json ToJson() const;
    static ImageAnnotations FromJson(const nlohmann::json& j);
};

/**
 * All annotations for a dataset
 */
struct AnnotationSet {
    std::string dataset_id;
    std::vector<std::string> class_names;  // class_id -> name (index 0 = class 0)
    std::map<size_t, ImageAnnotations> images;  // image_idx -> annotations

    // Class management
    int AddClass(const std::string& name);  // Returns class_id
    int GetClassId(const std::string& name) const;  // Returns -1 if not found
    const std::string& GetClassName(int class_id) const;

    // Image annotation access
    ImageAnnotations& GetOrCreateImage(size_t image_idx, int width, int height);
    ImageAnnotations* GetImage(size_t image_idx);
    const ImageAnnotations* GetImage(size_t image_idx) const;

    // Statistics
    size_t GetTotalAnnotationCount() const;
    size_t GetAnnotatedImageCount() const;

    // JSON serialization
    nlohmann::json ToJson() const;
    static AnnotationSet FromJson(const nlohmann::json& j);
};

/**
 * Annotation Manager
 *
 * Central manager for all annotations across all datasets.
 * Accessed via DataRegistry::GetAnnotationManager()
 *
 * Features:
 * - Per-dataset, per-image annotation storage
 * - Class label management
 * - Export to COCO JSON, YOLO txt, Pascal VOC XML
 * - Persistence with project files
 * - Training integration (segmentation masks)
 */
class AnnotationManager {
public:
    AnnotationManager() = default;
    ~AnnotationManager() = default;

    // Non-copyable
    AnnotationManager(const AnnotationManager&) = delete;
    AnnotationManager& operator=(const AnnotationManager&) = delete;

    // =========================================================================
    // Dataset-level operations
    // =========================================================================

    /**
     * Create annotation set for a dataset (if not exists)
     * @param dataset_id Unique dataset identifier
     */
    void CreateAnnotationSet(const std::string& dataset_id);

    /**
     * Get annotation set for a dataset
     * @param dataset_id Dataset identifier
     * @return Pointer to annotation set, or nullptr if not found
     */
    AnnotationSet* GetAnnotationSet(const std::string& dataset_id);
    const AnnotationSet* GetAnnotationSet(const std::string& dataset_id) const;

    /**
     * Remove annotation set for a dataset
     * @param dataset_id Dataset identifier
     * @return true if removed, false if not found
     */
    bool RemoveAnnotationSet(const std::string& dataset_id);

    /**
     * Check if dataset has annotation set
     */
    bool HasAnnotationSet(const std::string& dataset_id) const;

    // =========================================================================
    // Class management
    // =========================================================================

    /**
     * Add class to dataset (if not exists)
     * @param dataset_id Dataset identifier
     * @param class_name Name of the class
     * @return Class ID (index in class list)
     */
    int AddClass(const std::string& dataset_id, const std::string& class_name);

    /**
     * Get all class names for a dataset
     * @param dataset_id Dataset identifier
     * @return Reference to class names vector
     */
    const std::vector<std::string>& GetClasses(const std::string& dataset_id) const;

    // =========================================================================
    // Annotation operations
    // =========================================================================

    /**
     * Add annotation to an image
     * @param dataset_id Dataset identifier
     * @param image_idx Image index in dataset
     * @param class_id Class ID
     * @param type Annotation type
     * @param polygon Polygon points (for Polygon type)
     * @param bbox Bounding box (for BBox type)
     * @param image_width Image width (for RLE encoding)
     * @param image_height Image height (for RLE encoding)
     * @return Annotation ID, or -1 on error
     */
    int AddAnnotation(const std::string& dataset_id, size_t image_idx,
                      int class_id, AnnotationType type,
                      const std::vector<Point2D>& polygon,
                      const BoundingBox& bbox = BoundingBox{},
                      int image_width = 0, int image_height = 0);

    /**
     * Remove annotation from an image
     * @param dataset_id Dataset identifier
     * @param image_idx Image index
     * @param ann_id Annotation ID
     * @return true if removed
     */
    bool RemoveAnnotation(const std::string& dataset_id, size_t image_idx, int ann_id);

    /**
     * Get annotations for an image
     * @param dataset_id Dataset identifier
     * @param image_idx Image index
     * @return Pointer to annotation list, or nullptr
     */
    std::vector<Annotation>* GetAnnotations(const std::string& dataset_id, size_t image_idx);
    const std::vector<Annotation>* GetAnnotations(const std::string& dataset_id, size_t image_idx) const;

    /**
     * Clear all annotations for an image
     */
    void ClearImageAnnotations(const std::string& dataset_id, size_t image_idx);

    // =========================================================================
    // Export functions
    // =========================================================================

    /**
     * Export annotations to COCO JSON format
     * @param dataset_id Dataset identifier
     * @param output_path Output JSON file path
     * @param image_dir Directory containing images (for file_name field)
     * @return true on success
     */
    bool ExportCOCO(const std::string& dataset_id, const std::string& output_path,
                    const std::string& image_dir = "") const;

    /**
     * Export annotations to YOLO txt format
     * Creates one .txt file per image with annotations
     * @param dataset_id Dataset identifier
     * @param output_dir Output directory
     * @param classes_file If not empty, also writes classes.txt
     * @return true on success
     */
    bool ExportYOLO(const std::string& dataset_id, const std::string& output_dir,
                    const std::string& classes_file = "") const;

    /**
     * Export annotations to Pascal VOC XML format
     * Creates one .xml file per image
     * @param dataset_id Dataset identifier
     * @param output_dir Output directory
     * @param image_dir Directory containing images (for path field)
     * @return true on success
     */
    bool ExportVOC(const std::string& dataset_id, const std::string& output_dir,
                   const std::string& image_dir = "") const;

    // =========================================================================
    // Training integration
    // =========================================================================

    /**
     * Generate segmentation mask from annotations
     * @param dataset_id Dataset identifier
     * @param image_idx Image index
     * @param width Output mask width
     * @param height Output mask height
     * @return Mask with class IDs per pixel (0 = background)
     */
    std::vector<float> GetSegmentationMask(const std::string& dataset_id,
                                            size_t image_idx,
                                            int width, int height) const;

    /**
     * Generate binary mask for a single class
     * @param dataset_id Dataset identifier
     * @param image_idx Image index
     * @param class_id Class to extract (all others = 0)
     * @param width Output mask width
     * @param height Output mask height
     * @return Binary mask (0 or 1)
     */
    std::vector<float> GetClassMask(const std::string& dataset_id,
                                     size_t image_idx, int class_id,
                                     int width, int height) const;

    // =========================================================================
    // Persistence (called by ProjectManager)
    // =========================================================================

    /**
     * Serialize all annotations to JSON
     */
    nlohmann::json ToJson() const;

    /**
     * Load annotations from JSON
     */
    void FromJson(const nlohmann::json& j);

    /**
     * Clear all annotations
     */
    void Clear();

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * Get total number of annotations across all datasets
     */
    size_t GetTotalAnnotationCount() const;

    /**
     * Get list of all dataset IDs with annotations
     */
    std::vector<std::string> GetDatasetIds() const;

private:
    std::map<std::string, AnnotationSet> annotation_sets_;

    // Empty class list for returning when dataset not found
    static const std::vector<std::string> empty_class_list_;
};

// Enum string conversions
std::string AnnotationTypeToString(AnnotationType type);
AnnotationType StringToAnnotationType(const std::string& str);

} // namespace cyxwiz
