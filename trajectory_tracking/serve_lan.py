"""Serve only the trajectory report assets, with HTTP Range for video seeking."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import mimetypes
import re
from urllib.parse import unquote, urlsplit

FILES={'index.html','rollout.mp4','robot.mp4','robot-preview.jpg','preview.png','tracking.png','motion_50hz.npz','motion_arms_50hz.npz','tokens_50hz.npz','summary.json','arm_export.json','episode.npz','batch.json','batch.csv','batch.png'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--bind',required=True)
    parser.add_argument('--port',type=int,default=19086)
    args=parser.parse_args()
    root=args.directory.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):self.respond(head=True)
        def do_GET(self):self.respond(head=False)

        def respond(self,head):
            name=unquote(urlsplit(self.path).path)
            if name=='/':name='/index.html'
            if name=='/previous/':name='/previous/index.html'
            if name=='/waypoints/':name='/waypoints/index.html'
            if name=='/precision/':name='/precision/index.html'
            if name=='/precision-extra10/':name='/precision-extra10/index.html'
            if name=='/different-paths/':name='/different-paths/index.html'
            if name=='/robot-walking/':name='/robot-walking/index.html'
            if name=='/decoder-replay/':name='/decoder-replay/index.html'
            parts=name.strip('/').split('/')
            allowed=(len(parts)==1 and parts[0] in FILES) or (len(parts)==2 and
                    re.fullmatch(r'trial-[0-9]+|previous|waypoints|precision|precision-extra10|different-paths|robot-walking|decoder-replay',parts[0]) and parts[1] in FILES) or (
                    len(parts)==3 and parts[0] in ['waypoints','precision','precision-extra10','different-paths','robot-walking'] and
                    re.fullmatch(r'trial-[0-9]+|previous',parts[1]) and parts[2] in FILES)
            if not allowed:
                self.send_error(404);return
            path=root/name[1:]
            if not path.is_file() or path.is_symlink():
                self.send_error(404);return
            size=path.stat().st_size
            start,end=0,size-1
            partial=False
            requested=self.headers.get('Range')
            if requested:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested.strip())
                if not match or not any(match.groups()):
                    self.send_error(416);return
                a,b=match.groups()
                if a:
                    start=int(a);end=min(int(b),size-1) if b else size-1
                else:
                    start=max(0,size-int(b))
                if start>end or start>=size:
                    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}')
                    self.send_header('Content-Length','0');self.end_headers();return
                partial=True
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(end-start+1))
            self.send_header('Accept-Ranges','bytes')
            self.send_header('Cache-Control','no-cache' if name.endswith('/index.html') else 'public, max-age=3600')
            self.send_header('X-Content-Type-Options','nosniff')
            if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if head:return
            try:
                with path.open('rb') as stream:
                    stream.seek(start);remaining=end-start+1
                    while remaining:
                        chunk=stream.read(min(1024*1024,remaining))
                        if not chunk:break
                        self.wfile.write(chunk);remaining-=len(chunk)
            except (BrokenPipeError,ConnectionResetError):pass

    server=ThreadingHTTPServer((args.bind,args.port),Handler)
    print(f'SONIC report: http://{args.bind}:{args.port}/',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
