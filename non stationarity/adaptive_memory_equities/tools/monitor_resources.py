"""External, low-overhead telemetry for an owned local run. Never changes models/data."""
import argparse
import json
import os
import time
from pathlib import Path
import psutil

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pid',required=True,type=int)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--interval',type=float,default=5)
args=parser.parse_args()
process=psutil.Process(args.pid)
identity=process.create_time()
args.output.parent.mkdir(parents=True,exist_ok=True)
process.cpu_percent();psutil.cpu_percent()
previous_disk=psutil.disk_io_counters();previous_time=time.monotonic()
samples=[]
with args.output.open('a') as stream:
 while process.is_running():
  try:
   if process.create_time()!=identity or process.status()==psutil.STATUS_ZOMBIE:break
   time.sleep(args.interval)
   now=time.monotonic();disk=psutil.disk_io_counters();memory=psutil.virtual_memory();swap=psutil.swap_memory()
   entry=dict(time=time.time(),pid=args.pid,process_cpu_percent=process.cpu_percent(),system_cpu_percent=psutil.cpu_percent(),
       process_rss_mib=process.memory_info().rss/2**20,process_threads=process.num_threads(),
       available_mib=memory.available/2**20,swap_used_mib=swap.used/2**20,load_average=os.getloadavg(),
       system_disk_read_mib_s=(disk.read_bytes-previous_disk.read_bytes)/2**20/(now-previous_time),
       system_disk_write_mib_s=(disk.write_bytes-previous_disk.write_bytes)/2**20/(now-previous_time),
       disk_free_gib=psutil.disk_usage(args.output.parent).free/2**30,
       memory_floor_breached=memory.available<250*2**20)
   stream.write(json.dumps(entry)+'\n');stream.flush();samples.append(entry)
   previous_disk,previous_time=disk,now
  except(psutil.NoSuchProcess,psutil.AccessDenied):break
if samples:
 summary=dict(samples=len(samples),interval_seconds=args.interval,logical_cpus=psutil.cpu_count(),
     process_cpu_mean=sum(x['process_cpu_percent'] for x in samples)/len(samples),
     process_cpu_max=max(x['process_cpu_percent'] for x in samples),
     system_cpu_max=max(x['system_cpu_percent'] for x in samples),
     peak_rss_mib=max(x['process_rss_mib'] for x in samples),
     minimum_available_mib=min(x['available_mib'] for x in samples),
     maximum_swap_mib=max(x['swap_used_mib'] for x in samples),
     minimum_disk_free_gib=min(x['disk_free_gib'] for x in samples),
     process_cpu_units='100 percent equals one logical core; machine has eight',
     disk_scope='whole system, not attributable solely to this process')
 args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
