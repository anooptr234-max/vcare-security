#!/usr/bin/env python3
"""Download Google Drive files, handling the virus-scan confirm page."""
import re
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def download(file_id, out_path):
    def get(url):
        req = urllib.request.Request(url, headers=UA)
        return urllib.request.urlopen(req, timeout=120)

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = get(url)
    ctype = resp.headers.get("Content-Type", "")
    if "text/html" in ctype:
        html = resp.read().decode("utf-8", "ignore")
        fields = dict(re.findall(
            r'<input type="hidden" name="([^"]+)" value="([^"]*)"', html))
        if "uuid" not in fields:
            # maybe a direct download link instead
            m = re.search(r'href="(/uc\?export=download[^"]+)"', html)
            if not m:
                raise RuntimeError("no confirm form found")
            url = "https://drive.google.com" + m.group(1).replace("&amp;", "&")
            resp = get(url)
            ctype = resp.headers.get("Content-Type", "")
            if "text/html" in ctype:
                raise RuntimeError("still HTML after confirm link")
        else:
            q = "&".join(f"{k}={v}" for k, v in fields.items())
            resp = get(f"https://drive.usercontent.google.com/download?{q}")
    total = 0
    with open(out_path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    print(f"{out_path}: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    jobs = [
        ("1zpyuacc96ojdGq6QYAfRDkBBm7MszK5_", "ppe_video1.mp4"),
        ("1wh6DOSX_6LigiOl9VAAaNetjek2ss8pu", "ppe_video2.mp4"),
        ("1l-ndgsMyRphSyFdDI8P2HE_6fQgtcjJ4", "ppe_video3.mp4"),
    ]
    for fid, out in jobs:
        try:
            download(fid, out)
        except Exception as e:
            print(f"FAILED {fid}: {e}")
