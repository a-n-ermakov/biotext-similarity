import os
import shutil
import glob
import subprocess
import threading
import datetime
import time
import re
from collections import defaultdict

from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from page_reader import PageReader
from bio_engine import SingleBioModel
from graph_engine import GraphEngine


WORK_DIR = os.environ['WORK_DIR']
WORK_DIR_IN = os.path.join(WORK_DIR, 'in')
WORK_DIR_OUT = os.path.join(WORK_DIR, 'out')
WORK_DIR_LAUNCH = os.path.join(WORK_DIR_OUT, 'launch')
WORK_DIR_IMG = os.path.join(WORK_DIR_OUT, 'img')
WORK_DIR_OCR = os.path.join(WORK_DIR_OUT, 'ocr')

SAMPLE_LINK = 'sample'
SAMPLE_IMG_LINK = 'sample_img'
SAMPLE_OCR_LINK = 'sample_ocr'
SAMPLE_META_NAME = 'sample_meta'

REF_LINK = 'ref'
REF_IMG_LINK = 'ref_img'
REF_OCR_LINK = 'ref_ocr'
REF_META_NAME = 'ref_meta'

STATUS_FILE = 'status'
RESULT_NAME = 'coloration'
RESULT_DIAG_NAME = 'cluster_diagram.png'

PARAM_SAMPLE = SAMPLE_LINK
PARAM_REF = REF_LINK
PARAMS_REQ = [PARAM_SAMPLE, PARAM_REF]

_FINISH = False  # stop signal for dispatcher


class Status(Enum):
    NEW = 1
    IMG = 2
    OCR = 3
    ANALYZE = 4
    END = 5

    @classmethod
    def has_name(cls, name):
        return any(name == item.name for item in cls)

class StatusHandler:
    
    def read(self, launch_dir):
        status_fn = os.path.join(launch_dir, STATUS_FILE)
        if not os.path.exists(status_fn):
            return Status.END
        with open(status_fn, 'r') as f:
            name = f.readline()
        if Status.has_name(name):
            return Status[name]
        else:
            return None
            
    def write(self, launch_dir, status):
        status_fn = os.path.join(launch_dir, STATUS_FILE)
        if status is Status.END:
            os.remove(status_fn)
        else:    
            with open(status_fn, 'w') as f:
                f.write(status.name)


class MetaHandler:

    def _parse(self, meta):
        page_metas = defaultdict(list)
        for s in meta.split('\n'):
            parts = re.split(' +', s.strip())
            if parts[0][0].isdigit():
                page_num = int(parts[0])
                size = int(parts[3]), int(parts[4])
                ppi = int(parts[12]), int(parts[13])
                page_metas[page_num] += [(size, ppi)]
        return page_metas            
    
    def read(self, launch_dir, meta_name):
        meta_fn = os.path.join(launch_dir, meta_name)
        if not os.path.exists(meta_fn):
            return None
        with open(meta_fn, 'r') as f:
            meta = f.read()
        return self._parse(meta)
    
    def write(self, launch_dir, meta_name, meta):
        meta_fn = os.path.join(launch_dir, meta_name)
        print(f'----- META -----\n{meta}\n----- META -----')
        with open(meta_fn, 'w') as f:
            f.write(meta)


class HttpHandler(BaseHTTPRequestHandler):

    def _set_headers(self, response_code=200):
        self.send_response(response_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        if self.path == 'tasks':
            self._set_headers()
            # todo show tasks with statuses ordered by datetime desc
            self.wfile.write("<html><body><h1>Tasks list</h1></body></html>")

    def do_HEAD(self):
        self._set_headers()
        
    def _save_meta(self, launch_dir, fn_link, meta_name):
        cmd = f'pdfimages -list {fn_link}'
        res = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
        meta = res.stdout.decode('utf-8').strip()
        MetaHandler().write(launch_dir, meta_name, meta)

    def _prepare_doc(self, launch_dir, fn_full, meta_name, fn_link_name, img_link_name, ocr_link_name):
        fn_link = os.path.join(launch_dir, fn_link_name)
        os.symlink(fn_full, fn_link)
        fn = fn_full.split('/')[-1]
        img_dir = os.path.join(WORK_DIR_IMG, fn)
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        img_link = os.path.join(launch_dir, img_link_name)
        os.symlink(img_dir, img_link)
        ocr_dir = os.path.join(WORK_DIR_OCR, fn)
        if not os.path.exists(ocr_dir):
            os.mkdir(ocr_dir)
        ocr_link = os.path.join(launch_dir, ocr_link_name)
        os.symlink(ocr_dir, ocr_link)
        self._save_meta(launch_dir, fn_link, meta_name)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/start':
            print('Start task request received')
            query_params = parse_qs(urlparse(self.path).query)
            if any([x not in PARAMS_REQ for x in query_params]):
                self._set_headers(400)
                self.wfile.write(f'Required query params {REQ_PARAMS} not defined'.encode('utf-8'))
                return
            sample_fn = os.path.join(WORK_DIR_IN, query_params[PARAM_SAMPLE][0])
            ref_fn = os.path.join(WORK_DIR_IN, query_params[PARAM_REF][0])
            if not os.path.exists(sample_fn) or not os.path.exists(ref_fn):
                self._set_headers(400)
                self.wfile.write(f'File {sample_fn} or {ref_fn} not found, please upload them into {WORK_DIR_IN}'.encode('utf-8'))
                return
            print(f'Start task request received for sample {sample_fn} and ref {ref_fn}')
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            launch_dir = os.path.join(WORK_DIR_LAUNCH, timestamp)
            if os.path.exists(launch_dir):
                self._set_headers(400)
                self.wfile.write(f'Launch dir [{launch_dir}] already exists')
                return
            os.mkdir(launch_dir)
            print(f'Task launched in [{launch_dir}]')
            self._prepare_doc(launch_dir, sample_fn, SAMPLE_META_NAME, SAMPLE_LINK, SAMPLE_IMG_LINK, SAMPLE_OCR_LINK)
            self._prepare_doc(launch_dir, ref_fn, REF_META_NAME, REF_LINK, REF_IMG_LINK, REF_OCR_LINK)
            StatusHandler().write(launch_dir, Status.NEW)
            self._set_headers()
            # todo redirect to tasks
            self.wfile.write(f'<html><body><h1>Task launched in {launch_dir}</h1></body></html>'.encode('utf-8'))
        else:
            self._set_headers(404)


def _start_daemon(target, args):
    t = threading.Thread(target=target, args=args)
    t.daemon = True
    t.start()
    return t

    
class Dispatcher:
    
    IMG_MAX_WIDTH = 2000
    
    status_handler = StatusHandler()
    meta_handler = MetaHandler()
    
    def __init__(self, timeout_sec=5):
        self.timeout_sec = timeout_sec
        # check work dirs
        if not os.path.exists(WORK_DIR_IN):
            os.mkdir(WORK_DIR_IN)
        if not os.path.exists(WORK_DIR_OUT):
            os.mkdir(WORK_DIR_OUT)
        if not os.path.exists(WORK_DIR_LAUNCH):
            os.mkdir(WORK_DIR_LAUNCH)
        if not os.path.exists(WORK_DIR_IMG):
            os.mkdir(WORK_DIR_IMG)
        if not os.path.exists(WORK_DIR_OCR):
            os.mkdir(WORK_DIR_OCR)
    
    def _dispatch_img(self, fn_link, img_dir, meta, skip_exist=True, use_gs=True):
        page_count = len(meta)
        for page_num in range(1, page_count + 1):
            page_idx = page_num - 1
            page_path = os.path.join(img_dir, f'page-{page_idx:03}.png')
            if skip_exist and os.path.exists(page_path):
                continue
            page_meta = meta[page_num]
            single_image = len(page_meta) == 1
            if single_image:
                size, ppi = page_meta[0]
            else:
                sizes = [size for size, ppi in page_meta]
                size = max(sizes)
                ppi = [ppi for size, ppi in page_meta][sizes.index(size)]
            width, height = size
            k = width / self.IMG_MAX_WIDTH
            need_scale = k > 1
            ppi_x, ppi_y = ppi
            if need_scale:
                ppi_x, ppi_y = int(ppi_x / k), int(ppi_y / k)
            print(size, width, height, k, ppi, ppi_x, ppi_y)
                
            if use_gs:
                cmd = 'gs -q -sDEVICE=pnggray -dSAFER -dBATCH -dNOPAUSE -dNumRenderingThreads=4 ' + \
                      f'-dFirstPage={page_num} -dLastPage={page_num} -r{ppi_x}x{ppi_y} -o{page_path} {fn_link}'
                print('CMD:', cmd)
                proc = subprocess.run(cmd, shell=True)
            else:    
                curpage_path = os.path.join(img_dir, 'curpage')
                if single_image and not need_scale:
                    cmd = f'pdfimages -f {page_num} -l {page_num} -png {fn_link} {curpage_path}'
                    curpage_path = os.path.join(img_dir, 'curpage-000.png')
                else:
                    cmd = f'pdftoppm -f {page_num} -l {page_num} -png -rx {ppi_x} -ry {ppi_y} {fn_link} {curpage_path}'
                    curpage_path = os.path.join(img_dir, f'curpage-{page_num:03}.png')
                print('CMD:', cmd)
                proc = subprocess.Popen(cmd, shell=True)
                proc.wait()
                shutil.move(curpage_path, page_path)
                cmd = f'convert -grayscale average {page_path} {page_path}'
                print('CMD:', cmd)
                proc = subprocess.Popen(cmd, shell=True)
                proc.wait()
        
    def _check_files_count(self, files_dir, ext, count):
        files_mask = os.path.join(files_dir, f'page-*.{ext}')
        files = glob.glob(files_mask)
        if 0 <= len(files) < count:
            return False
        last_file = files[-1]
        delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(last_file))
        return delta.seconds > 30
    
    def _dispatch_ocr(self, img_dir, ocr_dir, skip_exist=True):
        images = os.listdir(img_dir)
        page_count = len(images)
        for page_idx in range(page_count):
            page_path = os.path.join(ocr_dir, f'page-{page_idx:03}.hocr')
            if skip_exist and os.path.exists(page_path):
                continue
            fn_ocr = os.path.join(ocr_dir, 'page-%03d' % page_idx)
            fn_img = os.path.join(img_dir, 'page-%03d.png' % page_idx)
            cmd = f'tesseract {fn_img} {fn_ocr} -l rus+eng --oem 1 hocr'
            print('CMD:', cmd)
            subprocess.run(cmd, shell=True)

    def _dispatch_ocr_all(self, sample_img_link, sample_ocr_link, ref_img_link, ref_ocr_link):
        self._dispatch_ocr(sample_img_link, sample_ocr_link)
        self._dispatch_ocr(ref_img_link, ref_ocr_link)

    def _dispatch_process(self, sample_img_link, sample_ocr_link, ref_img_link, ref_ocr_link, 
                          result_fn, result_diag_fn=None):
        sample_reader = PageReader(sample_ocr_link)
        sample_reader.read_pages()
        ref_reader = PageReader(ref_ocr_link)
        ref_reader.read_pages()
        model = SingleBioModel(sample_reader, ref_reader)
        model.fit_transform()
        model.predict(show=True, fn_clusters=result_diag_fn)
        graph_engine = GraphEngine(sample_reader, ref_reader, sample_img_link, ref_img_link)
        graph_engine.prepare_colorations(model.areas_df, result_fn)
    
    def _dispatch(self):
        pattern = os.path.join(WORK_DIR_LAUNCH, '*')
        heavy_task_running = False
        for launch_dir in sorted(glob.glob(pattern)):
            status = self.status_handler.read(launch_dir)
            if status in [Status.IMG, Status.OCR]:
                print('Heavy task execution detected')
                heavy_task_running = True
                break
        
        for launch_dir in sorted(glob.glob(pattern)):
            status = self.status_handler.read(launch_dir)
            if status is None or status is Status.END:
                continue
            print(f'launch {launch_dir} - {status}')
            sample_fn_link  = os.path.join(launch_dir, SAMPLE_LINK)
            sample_img_link = os.path.join(launch_dir, SAMPLE_IMG_LINK)
            sample_ocr_link = os.path.join(launch_dir, SAMPLE_OCR_LINK)
            ref_fn_link     = os.path.join(launch_dir, REF_LINK)
            ref_img_link    = os.path.join(launch_dir, REF_IMG_LINK)
            ref_ocr_link    = os.path.join(launch_dir, REF_OCR_LINK)
            result_fn       = os.path.join(launch_dir, RESULT_NAME)
            result_diag_fn  = os.path.join(launch_dir, RESULT_DIAG_NAME)
            if status is not Status.ANALYZE:
                sample_meta  = self.meta_handler.read(launch_dir, SAMPLE_META_NAME)
                sample_count = len(sample_meta)
                ref_meta     = self.meta_handler.read(launch_dir, REF_META_NAME)
                ref_count    = len(ref_meta)
            if status is Status.NEW:
                if not heavy_task_running:
                    heavy_tasks_running = True
                    self.status_handler.write(launch_dir,Status.IMG)
                    _start_daemon(self._dispatch_img, (sample_fn_link, sample_img_link, sample_meta))
                    _start_daemon(self._dispatch_img, (ref_fn_link, ref_img_link, ref_meta))
                    break
            elif status is Status.IMG:
                if self._check_files_count(sample_img_link, 'png', sample_count) \
                  and self._check_files_count(ref_img_link, 'png', ref_count):
                    self.status_handler.write(launch_dir, Status.OCR)
                    _start_daemon(self._dispatch_ocr_all, (sample_img_link, sample_ocr_link, 
                                                           ref_img_link, ref_ocr_link))
                    break
            elif status is Status.OCR:
                if self._check_files_count(sample_ocr_link, 'hocr', sample_count) \
                  and self._check_files_count(ref_ocr_link, 'hocr', ref_count):
                    self.status_handler.write(launch_dir, Status.ANALYZE)
                    _start_daemon(self._dispatch_process, (sample_img_link, sample_ocr_link, 
                                                           ref_img_link, ref_ocr_link,
                                                           result_fn, result_diag_fn))
            elif status is Status.ANALYZE:
                if os.path.exists(result_fn + '.pdf'):
                    self.status_handler.write(launch_dir, Status.END)

    def dispatch_forever(self):
        print(f'Starting dispatcher for directory [{WORK_DIR}] with timeout={self.timeout_sec}sec ...')
        while not _FINISH:
            print('=DISP=')
            self._dispatch()
            time.sleep(self.timeout_sec)
        print ('Stopping dispatcher ...')
            
            
def run(server_class=HTTPServer, handler_class=HttpHandler, port=8080):
    try:
        dispatcher = Dispatcher()
        t = _start_daemon(target=dispatcher.dispatch_forever, args=())

        server_address = ('', port)
        httpd = server_class(server_address, handler_class)
        print(f'Starting httpd on port {port} ...')
        httpd.serve_forever()
        
    except KeyboardInterrupt:
        global _FINISH
        _FINISH = True
        t.join()

        print ('Stopping httpd by ^C ...')
        httpd.socket.close()
        
    
if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()

