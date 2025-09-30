# facekit/postprocessing/post_pass_sweeper.py
from typing import Dict, List, Tuple
import numpy as np
import cv2

def is_orphan_candidate(global_id_to_tracks: Dict[int, List], track, match_meta, 
                        emb_count_min=2, len_min=12, low_conf_band: Tuple[float,float]=(0.32,0.38)) -> bool:
    gid = track.global_id
    # singleton?
    if gid is None or len(global_id_to_tracks.get(gid, [])) == 1:
        return True
    # low-confidence merge + thin
    score = match_meta.get(track.track_id, {}).get("cosine_dist", None)
    is_low_conf = (score is not None) and (low_conf_band[0] <= score <= low_conf_band[1])
    is_thin = (len(track.embeddings) < emb_count_min) and (track.length() >= len_min)
    return bool(is_low_conf and is_thin)

def robust_center(embs: np.ndarray, outlier_cosdist: float = 0.30) -> np.ndarray:
    embs = embs.astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
    center = embs.mean(axis=0)
    dists = 1.0 - (embs @ center / np.linalg.norm(center).clip(min=1e-9))
    keep = dists <= outlier_cosdist
    kept = embs[keep] if keep.any() else embs
    return kept.mean(axis=0)

def flow_align_at(frame_bgr, bbox, roi_landmarks, 
                  align_fn, lk_flow_fn, prev_gray_roi, 
                  blur_thresh: float = 60.0):
    x1,y1,x2,y2 = bbox
    curr_roi = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    new_lms, status = lk_flow_fn(prev_gray_roi, curr_roi, roi_landmarks)
    if int(status.sum()) < 4:
        return None, None, None
    # quick blur gate
    if cv2.Laplacian(curr_roi, cv2.CV_64F).var() < blur_thresh:
        return None, None, None
    full_lms = [(x1+int(px), y1+int(py)) for px,py in new_lms]
    aligned = align_fn(frame_bgr, full_lms, source="flow")
    if aligned is None:
        return None, None, None
    return aligned, curr_roi, new_lms

def backfill_embeddings_via_flow(track, frame_provider, need: int, 
                                 lk_flow_fn, align_fn, embed_fn, 
                                 max_samples: int = 6):
    if need <= 0:
        return []

    # choose up to max_samples frames, evenly spaced across [start,end]
    start, end = track.first_frame_idx, track.last_frame_idx
    total = min(max_samples, need)
    frames = np.linspace(start, end, num=total+2, dtype=int)[1:-1].tolist()  # avoid edges

    added = []
    prev_gray_roi = track.last_gray_roi
    roi_landmarks = track.last_landmarks
    bbox = track.last_bbox
    if prev_gray_roi is None or roi_landmarks is None or bbox is None:
        return added

    for f in frames:
        frame = frame_provider.get_frame(track.shot_id, f)
        if frame is None:
            continue
        aligned, new_roi, new_lms = flow_align_at(frame, bbox, roi_landmarks, 
                                                  align_fn, lk_flow_fn, prev_gray_roi)
        if aligned is None:
            continue
        emb = embed_fn([aligned])[0]
        track.embeddings.append(emb)
        # advance state for next propagation
        prev_gray_roi = new_roi
        roi_landmarks = new_lms
        added.append(emb)
        if len(added) >= need:
            break
    return added

def post_pass_sweeper(tracks: List, resolver, frame_provider, 
                      global_id_to_tracks: Dict[int, List], match_meta: Dict[str,dict],
                      N_min: int = 4, outlier_cosdist: float = 0.30,
                      max_samples: int = 6, enable_redetect: bool = False,
                      redetect_fn=None, pick_best_frame_fn=None):
    """
    tracks: flat list of FaceTrack across all shots
    resolver: GlobalIdentityResolver with .relink_track(track)
    frame_provider: FrameProvider to fetch frames
    global_id_to_tracks: mapping gid -> [tracks]
    match_meta: per-track diagnostics from first-pass matching (e.g., cosine_dist)
    """
    from facekit.utils.optical_flow import propagate_landmarks_via_optical_flow as lk_flow_fn
    from facekit.embedding.alignment import align_face_for_arcface as align_fn
    from facekit.embedding.embedder import FaceEmbedder
    embedder = FaceEmbedder.get_instance()

    rescued, linked = 0, 0
    for t in tracks:
        if not is_orphan_candidate(global_id_to_tracks, t, match_meta):
            continue

        # backfill to reach N_min
        need = max(0, N_min - len(t.embeddings))
        if need > 0:
            new_embs = backfill_embeddings_via_flow(
                t, frame_provider, need, lk_flow_fn, align_fn, embedder.get_embedding_batch, max_samples
            )
            rescued += 1 if new_embs else 0

        # recompute representative (robust)
        if t.embeddings:
            rep = robust_center(np.stack(t.embeddings), outlier_cosdist)
            t.representative_embedding = rep  # cache if this field present

        # retry linking
        if resolver.relink_track(t):
            linked += 1
            continue

        # (optional) escalate with one re-detect on a best frame
        if enable_redetect and redetect_fn and pick_best_frame_fn:
            best_f = pick_best_frame_fn(t)  # e.g., least blur mid-frame
            emb = redetect_fn(t, best_f)
            if emb is not None:
                t.embeddings.append(emb)
                rep = robust_center(np.stack(t.embeddings), outlier_cosdist)
                t.representative_embedding = rep
                if resolver.relink_track(t):
                    linked += 1

    return {"rescued_tracks": rescued, "linked_after_rescue": linked}
