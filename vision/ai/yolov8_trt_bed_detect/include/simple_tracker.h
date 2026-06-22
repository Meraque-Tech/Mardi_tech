#pragma once
#include <vector>
#include <map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "types.h"

struct TrackedObj {
    int                      id;
    int                      class_id;
    cv::Ptr<cv::TrackerMOSSE> tracker;
    cv::Rect2d               bbox;
    int                      lost;
};

class SimpleTracker {
public:
    SimpleTracker(float iou_thresh = 0.3f, int max_lost = 5)
        : iou_thresh_(iou_thresh), max_lost_(max_lost), next_id_(0) {}

    // Call each frame with the current image and detections.
    // Returns cumulative unique-object counts per class.
    std::map<int, int> update(const cv::Mat &frame, const std::vector<Detection> &dets) {
        // 1. Update all existing trackers with the new frame
        for (auto &t : tracks_) {
            if (!t.tracker->update(frame, t.bbox))
                t.lost++;
        }

        // 2. Match detections to updated tracker predictions via IoU
        std::vector<bool> det_matched(dets.size(), false);
        std::vector<bool> trk_matched(tracks_.size(), false);

        for (size_t d = 0; d < dets.size(); ++d) {
            cv::Rect2d det_rect = to_rect(dets[d].bbox, frame);
            float best_iou = iou_thresh_;
            int   best_t   = -1;

            for (size_t t = 0; t < tracks_.size(); ++t) {
                if (trk_matched[t]) continue;
                if (tracks_[t].class_id != static_cast<int>(dets[d].class_id)) continue;
                float iou = compute_iou(det_rect, tracks_[t].bbox);
                if (iou > best_iou) { best_iou = iou; best_t = t; }
            }

            if (best_t >= 0) {
                // re-initialize tracker at new detected bbox (more accurate than prediction)
                tracks_[best_t].bbox  = to_rect(dets[d].bbox, frame);
                tracks_[best_t].lost  = 0;
                tracks_[best_t].tracker = cv::TrackerMOSSE::create();
                tracks_[best_t].tracker->init(frame, tracks_[best_t].bbox);
                trk_matched[best_t]   = true;
                det_matched[d]        = true;
            }
        }

        // 3. Unmatched detections → new unique objects
        for (size_t d = 0; d < dets.size(); ++d) {
            if (det_matched[d]) continue;
            TrackedObj obj;
            obj.id       = next_id_++;
            obj.class_id = static_cast<int>(dets[d].class_id);
            obj.lost     = 0;
            obj.bbox     = to_rect(dets[d].bbox, frame);
            obj.tracker  = cv::TrackerMOSSE::create();
            obj.tracker->init(frame, obj.bbox);
            tracks_.push_back(obj);
            unique_counts_[obj.class_id]++;
        }

        // 4. Remove tracks lost too long
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
            [&](const TrackedObj &t) { return t.lost > max_lost_; }), tracks_.end());

        return unique_counts_;
    }

    void reset() { tracks_.clear(); unique_counts_.clear(); next_id_ = 0; }

    const std::vector<TrackedObj> &tracks() const { return tracks_; }

private:
    float iou_thresh_;
    int   max_lost_;
    int   next_id_;
    std::vector<TrackedObj> tracks_;
    std::map<int, int>      unique_counts_;

    // Detection bbox: cx, cy, w, h (normalised to input size) → cv::Rect2d in pixels
    static cv::Rect2d to_rect(const float *bbox, const cv::Mat &frame) {
        float x = (bbox[0] - bbox[2] / 2.f) * frame.cols;
        float y = (bbox[1] - bbox[3] / 2.f) * frame.rows;
        float w = bbox[2] * frame.cols;
        float h = bbox[3] * frame.rows;
        return cv::Rect2d(x, y, w, h);
    }

    static float compute_iou(const cv::Rect2d &a, const cv::Rect2d &b) {
        cv::Rect2d inter = a & b;
        float i = inter.area();
        float u = a.area() + b.area() - i;
        return u > 0 ? i / u : 0.f;
    }
};
