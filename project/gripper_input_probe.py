"""Read-only leader gripper direction probe; no follower or motor writes."""
import json
from pathlib import Path
import subprocess
import time
import uuid

from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig


def main():
    port='/dev/cu.usbmodem5B421342561'
    if subprocess.run(['lsof','-t',port],capture_output=True,text=True).stdout.strip():
        raise RuntimeError('Leader port is already in use')
    directory=Path(__file__).resolve().parent/'vision_state'
    path=directory/('gripper_input_'+time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.jsonl')
    robot=SO100Leader(SO100LeaderConfig(port=port,id='pawacha_so100_leader'))
    try:
        robot.bus.connect(handshake=False)
        with path.open('x',buffering=1) as output:
            print('READ ONLY: pinch closed, hold 2 seconds, then open and hold 2 seconds. Log: '+str(path),flush=True)
            end=time.monotonic()+120
            while time.monotonic()<end:
                row={'time':time.time()}
                for key in ('Present_Position','Torque_Enable','Status'):
                    row[key]=int(robot.bus.read(key,'gripper',normalize=False,num_retry=2))
                output.write(json.dumps(row)+'\n')
                if row['Torque_Enable'] or row['Status']:
                    raise RuntimeError('Leader gripper must remain passive without an alarm')
                time.sleep(.1)
        print('Probe complete; direction still requires matching the user-labeled closed/open intervals.',flush=True)
    finally:
        if robot.bus.is_connected:robot.bus.disconnect(disable_torque=False)


if __name__=='__main__':main()
