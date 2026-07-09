"""
从 PSS.json 裁剪所有 task 的视频 segment，保存为独立 mp4。
用法:
    python extract_pss_segments.py
"""
import json, os, subprocess, shutil, tempfile

PSS_JSON = "/media/data6/xuejj/AgentCVR/agent_system/question/PSS.json"
VIDEO_BASE = "/media/data6/xuejj/CrossVid/videos"
OUT_BASE = "/media/data6/xuejj/CrossVid/PSS_video"

# 只处理主日志中出错的 86 个 task
WRONG_IDS = {1, 4, 7, 8, 9, 15, 16, 19, 21, 23, 27, 28, 29, 30, 31, 34, 37, 40, 44,
             46, 47, 60, 64, 66, 67, 71, 72, 75, 78, 79, 82, 83, 85, 88, 89, 90, 91,
             94, 96, 98, 99, 103, 109, 110, 112, 114, 115, 118, 122, 123, 132, 134, 137,
             139, 140, 147, 150, 155, 156, 157, 169, 171, 177, 178, 179, 180, 181, 183,
             184, 186, 191, 195, 196, 198, 199, 205, 210, 211, 215, 216, 218, 219, 225,
             234, 235, 238}

with open(PSS_JSON) as f:
    all_tasks = json.load(f)

tasks = [t for t in all_tasks if t["id"] in WRONG_IDS]
print(f"共 {len(all_tasks)} 个 task，只提取 {len(tasks)} 个错误 case")
ok, skip, fail = 0, 0, 0

for task in tasks:
    tid = task["id"]
    video_path = os.path.join(VIDEO_BASE, task["video"])
    segments = task["segments"]

    if not os.path.exists(video_path):
        print(f"  [Task {tid}] ❌ 视频不存在: {video_path}")
        fail += len(segments)
        continue

    out_dir = os.path.join(OUT_BASE, f"Task_{tid}")
    os.makedirs(out_dir, exist_ok=True)

    for seg_num in sorted(segments.keys(), key=int):
        ranges = segments[seg_num]  # [[s1,e1], [s2,e2], ...]
        out_file = os.path.join(out_dir, f"{seg_num}.mp4")

        # 已存在则跳过
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            skip += 1
            continue

        if len(ranges) == 1:
            # 单段：-c copy 无损裁剪
            start, end = ranges[0]
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-i", video_path,
                "-t", str(end - start),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-loglevel", "error",
                out_file
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and os.path.getsize(out_file) > 0:
                ok += 1
            else:
                # -c copy 失败了回退到重新编码
                cmd2 = [
                    "ffmpeg", "-y",
                    "-ss", str(start), "-i", video_path,
                    "-t", str(end - start),
                    "-c:v", "libx264", "-c:a", "aac",
                    "-loglevel", "error",
                    out_file
                ]
                proc2 = subprocess.run(cmd2, capture_output=True, text=True)
                if proc2.returncode == 0:
                    ok += 1
                else:
                    print(f"  [Task {tid}] ❌ Seg {seg_num}: {proc2.stderr[:120]}")
                    fail += 1
        else:
            # 多段：精确 seek + re-encode 逐段裁剪到临时文件，再 concat 合并
            tmp_dir = tempfile.mkdtemp(prefix=f"pss_t{tid}_s{seg_num}_")
            tmp_files = []
            concat_ok = True

            for i, (s, e) in enumerate(ranges):
                tmp_file = os.path.join(tmp_dir, f"part_{i}.mp4")
                tmp_files.append(tmp_file)
                # 多段场景必须重新编码才能精确裁剪（-c copy 会对齐关键帧导致时长不准）
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(s), "-i", video_path,
                    "-t", str(e - s),
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-c:a", "aac",
                    "-loglevel", "error",
                    tmp_file
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0 or os.path.getsize(tmp_file) == 0:
                    print(f"  [Task {tid}] ❌ Seg {seg_num} part {i}: {proc.stderr[:120]}")
                    concat_ok = False
                    break

            if concat_ok:
                concat_list = os.path.join(tmp_dir, "concat.txt")
                with open(concat_list, "w") as f:
                    for tf in tmp_files:
                        f.write(f"file '{tf}'\n")

                cmd_concat = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list,
                    "-c", "copy",
                    "-loglevel", "error",
                    out_file
                ]
                proc_concat = subprocess.run(cmd_concat, capture_output=True, text=True)
                if proc_concat.returncode == 0 and os.path.getsize(out_file) > 0:
                    ok += 1
                else:
                    print(f"  [Task {tid}] ❌ Seg {seg_num} concat: {proc_concat.stderr[:120]}")
                    fail += 1

            shutil.rmtree(tmp_dir, ignore_errors=True)

print(f"\n完成！成功={ok}, 跳过(已存在)={skip}, 失败={fail}")
print(f"输出目录: {OUT_BASE}")
