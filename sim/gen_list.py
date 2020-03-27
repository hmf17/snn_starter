import os
import sys

root = sys.argv[1]

sets = os.listdir(os.path.join(root, 'final'))
with open(os.path.join(root, 'list.csv'), 'w') as f:
    for s in sets:
        frames = os.listdir(os.path.join(root, 'final', s))
        frame_num = 0
        for frame in frames:
            if "dvs" in frame:
                frame_num = frame_num + 1

        f.write('%s,%d\n' % (s, frame_num))

