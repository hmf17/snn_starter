import os
import time
import rosbag
import sys

bash_path = "./sim.sh"
tmp_path = "/tmp/out.bag"
root = sys.argv[1]
rate = 20

def bag2txt(bag_path, out_path):
    b = rosbag.Bag(bag_path)
    for i, (topic, msgs, t) in enumerate(b.read_messages(topics=['/cam0/events'])):
        with open(os.path.join(out_path, 'dvs_%04d' % (i+1)), 'w') as f:
            f.write('%d,%d\n' % (msgs.width, msgs.height))
            for e in msgs.events:
                f.write('%d,%d,%d,%s\n' % (e.ts.nsecs, e.x, e.y, e.polarity))
    b.close()

sets = os.listdir(root)
for s in sets:
    system_bash = "%s %s %d" % (bash_path, os.path.join(root, s), rate)
    print(system_bash)
    os.system(system_bash)
    time.sleep(10)
    bag2txt(tmp_path, os.path.join(root, s))