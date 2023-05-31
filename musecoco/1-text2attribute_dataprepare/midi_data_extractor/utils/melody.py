def judge_melody_track(pos_info):
    pitch_record = {}
    for pos_item in pos_info:
        insts_notes = pos_item[-1]
        if insts_notes is None:
            continue

        for inst_id in insts_notes:
            if inst_id not in pitch_record:
                pitch_record[inst_id] = (0, 0)
            for pitch, dur, vel in insts_notes[inst_id]:
                sum_pitch, num_notes = pitch_record[inst_id]
                pitch_record[inst_id] = (sum_pitch + pitch, num_notes + 1)

    if 128 in pitch_record:
        pitch_record.pop(128)

    items = sorted(pitch_record.items(), key=lambda x: x[1][0] / x[1][1], reverse=True)
    num_beats = len(pos_info) / 12
    if items[0][1][1] / num_beats >= 0.5:
        return items[0][0], items

    return None, items

