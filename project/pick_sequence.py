"""Submission-demo sequence and readiness checks, independent of hardware I/O."""
BODY = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
CELLS = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
REQUIRED = ('approach', 'grasp', 'grasp_closed', 'lift', 'place', 'retreat')


def build_sequence(document, cell):
    if cell not in CELLS:
        raise ValueError('Unknown cell')
    if document.get('robot_id') != 'pawacha_so100_follower':
        raise ValueError('Robot ID mismatch')
    poses = document.get('poses', {}).get(cell, {})
    missing = [name for name in REQUIRED if name not in poses]
    if missing:
        raise ValueError('ช่องนี้ยังขาดท่า: ' + ', '.join(missing))
    def body(name):
        return {m: int(poses[name]['raw_position'][m]) for m in BODY}
    opened = int(poses['grasp']['raw_position']['gripper'])
    closed = int(poses['grasp_closed']['raw_position']['gripper'])
    if closed >= opened:
        raise ValueError('ค่าท่าปิดต้องน้อยกว่าท่าเปิดตาม calibration ชุดนี้')
    outside, lifted = body('retreat'), body('lift')
    aligned = dict(lifted, shoulder_pan=outside['shoulder_pan'])
    steps = []
    def add(kind, label, target):
        steps.append({'kind': kind, 'label': label, 'target_raw': dict(target)})
    add('body', 'ท่าเริ่มนอกตาราง', outside)
    add('gripper', 'เปิดกริปเปอร์', {'gripper': opened})
    add('body', 'ยกระดับนอกตาราง', aligned)
    add('body', 'เคลื่อนเหนือช่องต้นทาง', lifted)
    add('body', 'เข้าหาวัตถุ', body('approach'))
    add('body', 'ลงตำแหน่งจับ', body('grasp'))
    add('gripper', 'ปิดกริปเปอร์เพื่อคีบ', {'gripper': closed})
    add('body', 'ยกพ้นจุดจับ', body('approach'))
    add('body', 'ยกพ้นตาราง', lifted)
    add('body', 'ย้ายออกนอกตาราง', aligned)
    add('body', 'ลงจุดวาง', body('place'))
    add('gripper', 'เปิดเพื่อปล่อย', {'gripper': opened})
    add('body', 'กลับท่าพัก', outside)
    return steps


def readiness(document, cell, snapshot):
    """Never infer scene validation, pickup success or safety from raw limits."""
    reasons = []
    try:
        steps = build_sequence(document, cell)
    except (ValueError, KeyError, TypeError) as exc:
        return {'ready': False, 'reasons': [str(exc)], 'steps': []}
    poses = document['poses'][cell]
    if not all(poses[name].get('validation', {}).get('ready_for_motion') is True
               for name in REQUIRED):
        reasons.append('ท่าที่บันทึกยังไม่ได้รับรองสำหรับการหยิบวัตถุจริงในฉากปัจจุบัน')
    if not snapshot:
        reasons.append('ยังไม่มีผลตรวจมอเตอร์ล่าสุด')
    else:
        if snapshot.get('error'):
            reasons.append('ตรวจมอเตอร์ไม่สำเร็จ: ' + snapshot['error'])
        if not snapshot.get('calibrated'):
            reasons.append('ยังยืนยัน calibration ไม่ได้')
        motors = snapshot.get('motors', {})
        for m in (*BODY, 'gripper'):
            row = motors.get(m)
            if row is None:
                reasons.append('ไม่มีข้อมูลมอเตอร์: ' + m)
                continue
            if row['Torque_Enable']:
                reasons.append('มีมอเตอร์เปิดแรงอยู่: ' + m)
            if row['Status'] or row['Operating_Mode'] != 0:
                reasons.append('สถานะหรือโหมดผิดปกติ: ' + m)
            if row['Present_Temperature'] >= 55:
                reasons.append('อุณหภูมิสูง: ' + m)
            if not row['Min_Voltage_Limit'] < row['Present_Voltage'] < row['Max_Voltage_Limit']:
                reasons.append('แรงดันนอกช่วงตั้งค่า: ' + m)
            lo, hi = row['Min_Position_Limit'], row['Max_Position_Limit']
            if not lo <= row['Present_Position'] <= hi:
                reasons.append('ตำแหน่งจริงอยู่นอกช่วง: ' + m)
            for step in steps:
                if m in step['target_raw'] and not lo <= step['target_raw'][m] <= hi:
                    reasons.append('เป้าหมายนอกช่วง: ' + step['label'] + '/' + m)
        first = steps[0]['target_raw']
        if any(m in motors and abs(motors[m]['Present_Position'] - first[m]) > 90 for m in BODY):
            reasons.append('ท่าปัจจุบันยังไม่ใกล้ท่าเริ่มที่บันทึกไว้ (เกินประมาณ 8°)')
    return {'ready': not reasons, 'reasons': reasons, 'steps': steps,
            'object_pickup_verified': False}
