// ---------------------------------------------------------------------------
// Project : Tracker
// File    : binding.cpp
// Purpose : pybind11 bindings for tracker_core (KalmanFilter, IMMFilter,
//           TrackerState) and tracker_gmc (GMCEstimator)
// Note   : pybind11 requires exceptions/RTTI; this is boundary code.
// ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include "association.h"
#include "kalman_filter.h"
#include "imm_filter.h"
#include "tracker_state.h"
#include "gmc_estimator.h"

namespace py = pybind11;

PYBIND11_MODULE(tracker_cpp, m) {
    m.doc() = "High-performance Kalman / IMM filter tracker (JSF C++ compliant)";

    m.attr("MAX_ASSOCIATION_CANDIDATES") = tracker::kMaxAssociationCandidates;

    m.def(
        "associate_single_track",
        [](
            py::array_t<float, py::array::c_style | py::array::forcecast> tracker_bbox,
            py::array_t<float, py::array::c_style | py::array::forcecast> detection_bboxes,
            py::object detection_scores_obj,
            float iou_threshold,
            float score_weight
        ) {
            const py::buffer_info tracker_info = tracker_bbox.request();
            const py::buffer_info det_info = detection_bboxes.request();

            if (tracker_info.ndim != 1 || tracker_info.shape[0] != tracker::kMeasDim) {
                throw py::value_error("tracker_bbox must have shape (4,)");
            }
            if (det_info.ndim != 2 || det_info.shape[1] != tracker::kMeasDim) {
                throw py::value_error("detection_bboxes must have shape (N, 4)");
            }
            if (det_info.shape[0] > tracker::kMaxAssociationCandidates) {
                throw py::value_error("detection_bboxes exceeds MAX_ASSOCIATION_CANDIDATES");
            }

            tracker::MeasVec tracker_box = tracker::MeasVec::Zero();
            auto* tracker_ptr = static_cast<float*>(tracker_info.ptr);
            for (int32_t i = 0; i < tracker::kMeasDim; ++i) {
                tracker_box(i) = tracker_ptr[i];
            }

            tracker::CandidateBoxArray detection_boxes{};
            tracker::CandidateScoreArray detection_scores{};
            auto* det_ptr = static_cast<float*>(det_info.ptr);
            const int32_t num_detections = static_cast<int32_t>(det_info.shape[0]);

            for (int32_t row = 0; row < num_detections; ++row) {
                detection_boxes[row] = tracker::MeasVec::Zero();
                for (int32_t col = 0; col < tracker::kMeasDim; ++col) {
                    detection_boxes[row](col) = det_ptr[(row * tracker::kMeasDim) + col];
                }
            }

            if (!detection_scores_obj.is_none()) {
                py::array_t<float, py::array::c_style | py::array::forcecast> detection_scores_arr =
                    detection_scores_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                const py::buffer_info score_info = detection_scores_arr.request();
                if (score_info.ndim != 1 || score_info.shape[0] != det_info.shape[0]) {
                    throw py::value_error("detection_scores must have shape (N,)");
                }
                auto* score_ptr = static_cast<float*>(score_info.ptr);
                for (int32_t idx = 0; idx < num_detections; ++idx) {
                    detection_scores[idx] = score_ptr[idx];
                }
            } else {
                for (int32_t idx = 0; idx < num_detections; ++idx) {
                    detection_scores[idx] = 0.0F;
                }
            }

            const tracker::AssociationResult result = tracker::associate_single_track(
                tracker_box,
                detection_boxes,
                detection_scores,
                num_detections,
                iou_threshold,
                score_weight
            );

            return py::make_tuple(
                result.matched_index,
                result.matched_iou,
                result.matched_score,
                result.matched_cost,
                result.has_match
            );
        },
        py::arg("tracker_bbox"),
        py::arg("detection_bboxes"),
        py::arg("detection_scores") = py::none(),
        py::arg("iou_threshold") = 0.3F,
        py::arg("score_weight") = 0.0F,
        "Associate one tracker bbox with up to MAX_ASSOCIATION_CANDIDATES detections."
    );

    // Explicit function pointer types for overload resolution (noexcept is
    // part of the function type in C++17).
    using kf_update_single_t =
        const tracker::StateVec& (tracker::KalmanFilter::*)(const tracker::MeasVec&) noexcept;
    using kf_update_adaptive_t =
        const tracker::StateVec& (tracker::KalmanFilter::*)(const tracker::MeasVec&, float) noexcept;

    using imm_update_single_t =
        const tracker::StateVec& (tracker::IMMFilter::*)(const tracker::MeasVec&) noexcept;
    using imm_update_adaptive_t =
        const tracker::StateVec& (tracker::IMMFilter::*)(const tracker::MeasVec&, float) noexcept;

    // ── KalmanFilter (10D, standalone CV) ───────────────────────
    py::class_<tracker::KalmanFilter>(m, "KalmanFilter")
        .def(py::init<>())
        .def("init", &tracker::KalmanFilter::init,
             py::arg("z0"),
             "Initialize state from first measurement [x, y, w, h]")
        .def("predict", &tracker::KalmanFilter::predict,
             py::return_value_policy::reference_internal,
             "Predict next state (coasting mode)")
        .def("update",
             static_cast<kf_update_single_t>(&tracker::KalmanFilter::update),
             py::arg("z"),
             py::return_value_policy::reference_internal,
             "Predict + correct with measurement [x, y, w, h]")
        .def("update",
             static_cast<kf_update_adaptive_t>(&tracker::KalmanFilter::update),
             py::arg("z"), py::arg("confidence"),
             py::return_value_policy::reference_internal,
             "Predict + correct with R scaled by 1 / max(conf, floor)")
        .def("apply_gmc", &tracker::KalmanFilter::apply_gmc,
             py::arg("H"),
             "Warp state mean by a 3x3 homography (prev -> curr)")
        .def("set_gmc_failed", &tracker::KalmanFilter::set_gmc_failed,
             py::arg("failed"),
             "Flag that GMC failed; next predict boosts Q then auto-clears")
        .def("set_gmc_q_boost", &tracker::KalmanFilter::set_gmc_q_boost,
             py::arg("boost"),
             "Set Q multiplier on GMC-failed frames (default 4.0)")
        .def("set_adaptive_r_floor", &tracker::KalmanFilter::set_adaptive_r_floor,
             py::arg("floor"),
             "Set confidence floor for adaptive R (default 0.4)")
        .def("get_state", &tracker::KalmanFilter::get_state,
             py::return_value_policy::reference_internal)
        .def("get_covariance", &tracker::KalmanFilter::get_covariance,
             py::return_value_policy::reference_internal)
        .def("is_initialized", &tracker::KalmanFilter::is_initialized)
        .def("reset", &tracker::KalmanFilter::reset)
        .def("set_process_noise", &tracker::KalmanFilter::set_process_noise,
             py::arg("Q"))
        .def("set_measurement_noise", &tracker::KalmanFilter::set_measurement_noise,
             py::arg("R"));

    // ── IMMFilter (3-model: CV + CA + Singer) ───────────────────
    py::class_<tracker::IMMFilter>(m, "IMMFilter")
        .def(py::init<>())
        .def("init", &tracker::IMMFilter::init,
             py::arg("z0"),
             "Initialize all 3 models from first measurement [x, y, w, h]")
        .def("predict", &tracker::IMMFilter::predict,
             py::return_value_policy::reference_internal,
             "IMM predict cycle (mix + per-model predict)")
        .def("update",
             static_cast<imm_update_single_t>(&tracker::IMMFilter::update),
             py::arg("z"),
             py::return_value_policy::reference_internal,
             "Full IMM cycle: mix, predict, likelihood, update, combine")
        .def("update",
             static_cast<imm_update_adaptive_t>(&tracker::IMMFilter::update),
             py::arg("z"), py::arg("confidence"),
             py::return_value_policy::reference_internal,
             "IMM update with R scaled by 1 / max(conf, floor)")
        .def("apply_gmc", &tracker::IMMFilter::apply_gmc,
             py::arg("H"),
             "Warp all per-model state means by a 3x3 homography")
        .def("set_gmc_failed", &tracker::IMMFilter::set_gmc_failed,
             py::arg("failed"),
             "Flag GMC failure; next predict boosts Q across all models")
        .def("set_gmc_q_boost", &tracker::IMMFilter::set_gmc_q_boost,
             py::arg("boost"),
             "Set Q multiplier on GMC-failed frames (default 4.0)")
        .def("set_adaptive_r_floor", &tracker::IMMFilter::set_adaptive_r_floor,
             py::arg("floor"),
             "Set confidence floor for adaptive R (default 0.4)")
        .def("get_state", &tracker::IMMFilter::get_state,
             py::return_value_policy::reference_internal,
             "Combined state estimate [x,y,w,h,vx,vy,vw,vh,ax,ay]")
        .def("get_covariance", &tracker::IMMFilter::get_covariance,
             py::return_value_policy::reference_internal,
             "Combined error-covariance (10x10)")
        .def("get_model_probabilities", &tracker::IMMFilter::get_model_probabilities,
             py::return_value_policy::reference_internal,
             "Model probabilities [mu_CV, mu_CA, mu_Singer]")
        .def("is_initialized", &tracker::IMMFilter::is_initialized)
        .def("reset", &tracker::IMMFilter::reset)
        .def("set_transition_matrix", &tracker::IMMFilter::set_transition_matrix,
             py::arg("pi"),
             "Override Markov transition matrix (3x3)")
        .def("set_model_process_noise", &tracker::IMMFilter::set_model_process_noise,
             py::arg("idx"), py::arg("Q"),
             "Override process noise Q for model idx (0=CV, 1=CA, 2=Singer)")
        .def("set_measurement_noise", &tracker::IMMFilter::set_measurement_noise,
             py::arg("R"),
             "Override measurement noise R (shared by all models)")
        .def("set_singer_params", &tracker::IMMFilter::set_singer_params,
             py::arg("alpha"), py::arg("sigma2_a"),
             "Configure Singer physics: alpha=1/tau (maneuver time reciprocal), "
             "sigma2_a=acceleration variance [px^2/frame^4]. Rebuilds F and Q immediately. "
             "alpha range: (0,20], sigma2_a range: (0,500].")
        .def("mahalanobis_sq",
             [](const tracker::IMMFilter& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> z_arr) -> float {
                 const py::buffer_info info = z_arr.request();
                 if (info.ndim != 1 || info.shape[0] != tracker::kMeasDim) {
                     throw py::value_error("z must have shape (4,)");
                 }
                 tracker::MeasVec z = tracker::MeasVec::Zero();
                 auto* ptr = static_cast<float*>(info.ptr);
                 for (int32_t i = 0; i < tracker::kMeasDim; ++i) {
                     z(i) = ptr[i];
                 }
                 return self.mahalanobis_sq(z);
             },
             py::arg("z"),
             "Mahalanobis distance^2 for measurement z=[x,y,w,h] against combined state. "
             "chi2(4dof) gates: p=0.10->7.78, p=0.01->13.28, p=0.001->18.47. "
             "Returns 1e4 when filter uninitialized or innovation covariance is singular.");

    // ── Model index constants ───────────────────────────────────
    m.attr("MODEL_CV")     = tracker::kModelCV;
    m.attr("MODEL_CA")     = tracker::kModelCA;
    m.attr("MODEL_SINGER") = tracker::kModelSinger;

    // ── TrackState enum ─────────────────────────────────────────
    py::enum_<tracker::TrackState>(m, "TrackState")
        .value("TRACKING", tracker::TrackState::TRACKING)
        .value("COASTING", tracker::TrackState::COASTING)
        .value("LOST", tracker::TrackState::LOST);

    // ── TrackerState ────────────────────────────────────────────
    py::class_<tracker::TrackerState>(m, "TrackerState")
        .def(py::init<>())
        .def("step", &tracker::TrackerState::step,
             py::arg("confidence"),
             "Advance state machine with AI confidence score")
        .def("force_tracking", &tracker::TrackerState::force_tracking)
        .def("state", &tracker::TrackerState::state)
        .def("coast_count", &tracker::TrackerState::coast_count)
        .def("set_confidence_threshold", &tracker::TrackerState::set_confidence_threshold,
             py::arg("threshold"))
        .def("set_max_coast_frames", &tracker::TrackerState::set_max_coast_frames,
             py::arg("n"));

    // ── GMCEstimator (ORB + partial-affine) ─────────────────────
    py::class_<tracker::GMCEstimator>(m, "GMCEstimator")
        .def(py::init<int32_t, int32_t, float, float, float, float>(),
             py::arg("n_features")           = tracker::kGmcDefaultFeatures,
             py::arg("min_matches")          = tracker::kGmcDefaultMinMatches,
             py::arg("inlier_ratio_thresh")  = tracker::kGmcDefaultInlierRatio,
             py::arg("ransac_reproj_thresh") = tracker::kGmcDefaultRansacReproj,
             py::arg("dilate_factor")        = tracker::kGmcDefaultDilateFactor,
             py::arg("downsample")           = tracker::kGmcDefaultDownsample)
        .def("estimate",
             [](tracker::GMCEstimator& self,
                py::array_t<uint8_t, py::array::c_style> prev_gray,
                py::array_t<uint8_t, py::array::c_style> curr_gray,
                py::array_t<float, py::array::c_style | py::array::forcecast> fg_xywh
             ) -> py::tuple {
                 const py::buffer_info p_info = prev_gray.request();
                 const py::buffer_info c_info = curr_gray.request();
                 const py::buffer_info f_info = fg_xywh.request();

                 if (p_info.ndim != 2 || c_info.ndim != 2) {
                     throw py::value_error("prev/curr must be 2-D grayscale");
                 }
                 if (f_info.ndim != 1 || f_info.shape[0] < 4) {
                     throw py::value_error("fg_xywh must have shape (4,)");
                 }

                 // Zero-copy cv::Mat wrappers over numpy buffers
                 cv::Mat p_mat(static_cast<int>(p_info.shape[0]),
                               static_cast<int>(p_info.shape[1]),
                               CV_8UC1, p_info.ptr);
                 cv::Mat c_mat(static_cast<int>(c_info.shape[0]),
                               static_cast<int>(c_info.shape[1]),
                               CV_8UC1, c_info.ptr);
                 auto* fg_ptr = static_cast<float*>(f_info.ptr);
                 float fg[4] = {fg_ptr[0], fg_ptr[1], fg_ptr[2], fg_ptr[3]};

                 tracker::HomMat H;
                 bool ok = self.estimate(p_mat, c_mat, fg, H);
                 return py::make_tuple(H, ok);
             },
             py::arg("prev_gray"),
             py::arg("curr_gray"),
             py::arg("fg_xywh"),
             "Estimate 3x3 affine-embedded homography (prev -> curr).");
}
