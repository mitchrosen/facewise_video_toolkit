def _dump_agg_state(label, agg, frame=None):
    print(f"\n===== {label} =====")
    if frame is not None:
        print(f"frame={frame}")
    open_tids = [t.track_id for t in agg.tracks if not t.is_closed()]
    print(f"open_tids={open_tids}")

    for t in agg.tracks:
        try:
            last_obs = t.observations[-1] if t.observations else None
            src = last_obs.source if last_obs is not None else None
            print(
                f"  tid={t.track_id}"
                f" first={t.first_frame()} last={t.last_frame()}"
                f" last_det={t.last_det_frame()}"
                f" closed={t.is_closed()}"
                f" bbox={t.get_last_bbox()}"
                f" src={src}"
            )
        except Exception as e:
            print(f"  tid={t.track_id} (ERROR dumping: {e})")
