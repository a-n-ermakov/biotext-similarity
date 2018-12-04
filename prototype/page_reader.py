import os
import glob
import re
from collections import defaultdict
from html.parser import HTMLParser
from page_reader_utils import find_occurences, clean_txt, extract_real_page_num, fix_real_page_nums
import numpy as np
from itertools import groupby


PAGE_NUM_STR_SIZE = 5
HYPHENS_ON_PAGE = 5

class _hOCR_Parser(HTMLParser):
    '''Custom parser for hOCR format from tesseract'''
    
    RE_COORDS = re.compile('bbox ((\d+ *?)+);')
    
    def __init__(self, page_idx, verbose=False):
        HTMLParser.__init__(self)
        self.page_idx = page_idx
        self.verbose = verbose
        self.ys = []
        self.lines = []
        self.__line = ''
        self.__cur_depth = 0
        self.__line_depth = None
    
    def handle_starttag(self, tag, attrs):
        self.__cur_depth += 1
        if tag == 'span':
            attrs_dict = {k: v for k, v in attrs}
            class_val, title_val = attrs_dict['class'], attrs_dict['title']
            if class_val == 'ocr_line':
                self.__line_depth = self.__cur_depth
                self.__line = ''
                if title_val is not None:
                    m = self.RE_COORDS.search(title_val)
                    if m is not None:
                        _, start, _, finish = tuple(m.group(1).split(' '))
                        start, finish = int(start), int(finish) 
                        self.ys.append((start, finish))
                    
    def handle_data(self, data):
        if self.__line_depth is not None:
            self.__line += data
    
    def handle_endtag(self, tag):
        if tag == 'span':
            if self.__line_depth == self.__cur_depth:
                # end of line detected
                self.__line_depth = None
                self.__line = self.__line.strip()
                if len(self.__line) > 0:
                    self.lines += [self.__line]
                else:
                    # ignoring empty lines - removing y
                    self.ys = self.ys[:-1]
                    return
        self.__cur_depth -= 1     

        #TODO check lines for ys (multiple spans on one `real` line)

def extract_page_nums(page_num_lines, verbose=False):
    '''Extract page nums from lines
    Input:
      page_num_lines:list(str) - strings with page nums
      verbose:bool - print verbose messages
    Output:
      list(int) - page nums
    '''
    page_nums = []
    for idx, line in enumerate(page_num_lines):
        page_num = extract_real_page_num(idx, line, verbose) if line is not None else None
        page_nums.append(page_num)
    return page_nums


def process_text(page_metas, page_num_on_top=True, word_hyphen=False, verbose=False):
    '''Process pages text
    Input:
      page_metas:list(dict(str,*)) - page metadatas (MODIFIED during process)
      page_num_on_top:bool - page number location (True - top, False - bottom, None - auto)
      word_hyphen:bool - use word hyphenation (None - auto)
      verbose:bool - print verbose messages
    '''
    # processing real page nums
    if verbose:
        print(f'PAGENUM on top: {page_num_on_top}')
    page_top_num_lines = [x['lines'][0] if len(x['lines']) > 0 else None for x in page_metas]
    page_bottom_num_lines = [x['lines'][-1] if len(x['lines']) > 0 else None for x in page_metas]
    if page_num_on_top is None:
        # detect page_num_on_top
        page_top_nums = extract_page_nums(page_top_num_lines)
        top_count = len([x for x in page_top_nums if x is not None])
        page_bottom_nums = extract_page_nums(page_bottom_num_lines)
        bottom_count = len([x for x in page_bottom_nums if x is not None])
        page_num_on_top = top_count > bottom_count
        page_nums = page_top_nums if page_num_on_top else page_bottom_nums
        if verbose:
            print(f'PAGENUM top {top_count}, bottom {bottom_count}')
            print(f'PAGENUM detected on {"top" if page_num_on_top else "bottom"}')
    elif page_num_on_top:
        page_nums = extract_page_nums(page_top_num_lines, verbose)
    else:
        page_nums = extract_page_nums(page_bottom_num_lines, verbose)
    
    if verbose:
        print(f'PAGENUM parsed real page nums: {page_nums}')

    # detect word_hyphen
    if verbose:
        print(f'WORD HYPHEN: {word_hyphen}')
    if word_hyphen is None:
        line_hyphens = [int(x.endswith('-')) for meta in page_metas for x in meta['lines']]
        hyphen_count = np.sum(line_hyphens)
        word_hyphen = hyphen_count > HYPHENS_ON_PAGE * len(page_metas)
        if verbose:
            print(f'WORD HYPHEN count: {hyphen_count}')
            print(f'WORD HYPHEN detected: {word_hyphen}')
        
    # final processing
    for idx, meta in enumerate(page_metas):
        # 1.saving real page num
        meta['real_page_num'] = page_nums[idx]
        # 2.removing real page num - for correct coloration (bad page number detection in images)
        lines = meta['lines']
        ys = meta['ys']
        if len(lines) == 0:
            print(f'WARN no lines found on page {idx}, skipping')
            continue
        page_line = lines[0] if page_num_on_top else lines[-1]
        if len(page_line) <= PAGE_NUM_STR_SIZE:
            ys = ys[1:] if page_num_on_top else ys[:-1]
            lines = lines[1:] if page_num_on_top else lines[:-1]
        # 3.word hyphens processing
        lines_prep = []
        for line in lines:
            if line.endswith('-'):
                line = line[:-1]
                if not word_hyphen:
                    line = line + ' '
            else:
                line = line + ' '
            lines_prep.append(line)
        # 4.cleaning lines
        lines = [clean_txt(line, multi_eol=False) for line in lines_prep]
        
        meta['ys'] = ys
        meta['lines'] = lines


# deprecated - bad practic results, image segmentation required
def extract_non_text(page_metas, verbose=False):
    '''Extract non text data from metadata (detect short line sequences)
    Input:
      page_metas:list(dict(str,*)) - page metadatas (MODIFIED during process)
      verbose:bool - print verbose messages
    Output:
      dict(int,list(tuple(int*4))) - dict page_idx->list of non-text object coordinates (left, up, right, bottom)
    '''
    images = {}
    SHORT_LINE_MAX_LENGTH = 30
    SHORT_LINES_MIN_COUNT = 3
    for page_idx, meta in enumerate(page_metas):
        lines = meta['lines']
        ys = meta['ys']
        # detecting short line ranges (inclusive)
        short_lines = [len(x) < SHORT_LINE_MAX_LENGTH for x in lines]
        groups = [(x, len(list(y))) for x, y in groupby(short_lines)]
        idx = 0
        image_lines = []
        txt_lines, txt_ys = [], []
        for short_line, count in groups:
            start, finish = idx, idx + count
            if not short_line or count < SHORT_LINES_MIN_COUNT:
                txt_lines += lines[start:finish]
                txt_ys += lines[start:finish]
            else:
                if verbose:
                    print(f'IMAGE_DETECT: page {page_idx}, lines {(start, finish)} - image detected')
                    for line in lines[start:finish]:
                        print(f'IMAGE_DETECT: page {page_idx} in line: [{line}]')
                image_lines.append((start, finish-1))
            idx += count
        if verbose and len(image_lines) > 0:
            print(f'IMAGE_DETECT: page {page_idx}, total {len(image_lines)} images detected')
        meta['lines'] = txt_lines
        meta['ys'] = txt_ys
        # extracting ys for images
        page_images = []
        for start, finish in image_lines:
            image_ys = ys[start][0], ys[finish][1]
            page_images.append(image_ys)
        meta['images'] = page_images
        images[page_idx] = page_images
        
    return images


def read_hocr_pages(path, skip_pages=None, page_num_on_top=True, word_hyphen=False, 
                    predef_real_page_nums=None, verbose=False):
    '''Reading and preprocessing text from hOCR files
    Input:
      path:str - path to files
      skip_pages:list(int) - pages to skip processing (i.e. title, contents, sources)
      page_num_on_top:bool - page number location (True - top, False - bottom, None - auto)
      word_hyphen:bool - use word hyphenation (None - auto)
      predef_real_page_nums:list(int) - predefined real page nums (None - automatic extraction)
      verbose:bool - print verbose messages
    Output:
      str - preprocessed text
      dict(int,tuple(int*2)) - dict of line start position->page coords(page idx, line on page idx) in preprocessed text
      dict(int,int) - dict of page file idx->real page numbers (from ocr)
      dict(int,list(tuple(int*2))) - dict page_idx->list of y coordinates of upper and lower edges of lines
    '''
    if verbose:
        print('Reading pages from [%s]' % path)

    files_path = os.path.join(path, 'page-*.hocr')
    filenames = sorted(glob.glob(files_path))
   
    if predef_real_page_nums is not None:
        assert len(predef_real_page_nums) == len(filenames), 'predefined real_page_nums count inconsistent with files count'

    page_metas = []
    for idx, fn in enumerate(filenames):
        with open(fn, 'r', encoding='utf-8') as f:
            # raw
            txt_html = f.read()
            # parse hOCR
            parser = _hOCR_Parser(idx, verbose)
            parser.feed(txt_html)
            # save meta
            page_metas.append({
                'lines': parser.lines,
                'ys': parser.ys
            })
            parser.close()

    process_text(page_metas, page_num_on_top, word_hyphen, verbose)
    ### extract_non_text(page_metas, verbose)
            
    txt_prep = ''
    line_ys = {}
    line_start_positions = {}
    real_page_nums = {}
    for idx, meta in enumerate(page_metas):
        # real page num
        real_page_nums[idx] = meta['real_page_num'] if predef_real_page_nums is None else predef_real_page_nums[idx]
        if skip_pages is not None and idx in skip_pages:
            continue
        # line ys    
        line_ys[idx] = meta['ys']
        # calculating line positions
        lines = meta['lines']
        line_starts = len(txt_prep) + np.cumsum([0] + [len(x) for x in lines[:-1]])
        for l, x in enumerate(line_starts):
            line_start_positions[x] = (idx, l)
        # text assembly    
        for line in lines:
            txt_prep += line
    # special pos for end of document                                                   
    line_start_positions[len(txt_prep)] = (None, None)

    # fixing real page nums
    if predef_real_page_nums is None:
        real_page_nums = fix_real_page_nums(real_page_nums, verbose)
    
    ### TODO some double spaces '  ' still remains in txt
    # print('PAGE_READER final txt:', txt_prep) 
    return txt_prep, line_start_positions, real_page_nums, line_ys

                
def convert_batch_positions_to_page_coords(positions, ls_positions):
    '''Convert positions to page coords using line start positions (for batch processing)
    Input:
      positions:list(int) - sample positions (sorted!)
      ls_positions:dict(int,tuple(int*2)) - dict of line start position->page coords(page idx, line on page idx)
    Output:  
      dict(int,tuple(int*2)) - dict of position->page coords(page idx, line on page idx)
      ### todo add (page idx, line on page idx) for end of sample
      ### todo maybe add 3rd coord - position of sample on line
    '''
    sorted_positions = sorted(ls_positions.keys())
    cur_ls_positions = iter(sorted_positions[:-1])
    next_ls_positions = iter(sorted_positions[1:])
    
    cur_ls_pos = next(cur_ls_positions)
    p, l = ls_positions[cur_ls_pos]
    next_ls_pos = next(next_ls_positions)

    page_coords = defaultdict(None)
    last_line = False
    for pos in positions:
        if not last_line:
            while not cur_ls_pos <= pos < next_ls_pos:
                try:
                    cur_ls_pos = next(cur_ls_positions)
                    p, l = ls_positions[cur_ls_pos]
                    next_ls_pos = next(next_ls_positions)
                except StopIteration:
                    last_line = True
                    print(f'WARN last line achieved for pos {pos}')
        # print('DEBUG pos', pos)
        page_coords[pos] = (p, l)
    return page_coords


def convert_position_to_page_coords(position, ls_positions):
    '''Convert position to page coords using line start positions - find nearest coords
    Input:
      position:int - sample position
      ls_positions:dict(int,tuple(int*2)) - dict of line start position->page coords(page idx, line on page idx)
    Output:  
      tuple(int*2) - page coords(page idx, line on page idx)
    '''
    while position not in ls_positions:
        position -= 1
    return ls_positions[position]


def convert_page_coords_to_position(page_coords, ls_positions_inv):
    '''Convert page coords to position using line start positions
    Input:
      page_coords:tuple(int*2) - page coords(page idx, line on page idx)
      ls_positions_inv:dict(tuple(int*2),int) - dict of page coords(page idx, line on page idx)->line start position
    Output:  
      int - sample position
    '''
    if page_coords not in ls_positions_inv:
        print('page coords not found:', page_coords)
        return None
    return ls_positions_inv[page_coords]


def convert_real_to_page_coords(real_page_coords, real_page_nums_inv):
    '''Convert `real` page coords to page coords
    Input:
      real_page_coords:tuple(int*2) - real page coords (page_number, line_number)
      real_page_nums_inv:dict(int,int) - dict of real page number->page file idx (from ocr)
    Output:
      tuple(int*2) - page coords (page idx, line on page idx) 
    '''
    page_num, line_num = real_page_coords
    if page_num not in real_page_nums_inv:
        print('real page number not found:', page_num)
        return None
    return real_page_nums_inv[page_num], line_num-1


def convert_page_to_real_coords(page_coords, real_page_nums):
    '''Convert page coords to `real` page coords
    Input:
      page_coords:tuple(int*2) - page coords (page idx, line on page idx)
      real_page_nums:dict(int,int) - dict of page file idx->real page number (from ocr)
    Output:
      tuple(int*2) - real page coords (page_number, line_number)
    '''
    page_idx, line_idx = page_coords
    if page_idx not in real_page_nums:
        print('page idx not found:', page_idx)
        return None
    return real_page_nums[page_idx], line_idx+1


class PageReader():
    '''Page reader wrapper'''
    
    def __init__(self, path, 
                 skip_pages=None, 
                 page_num_on_top=True, 
                 word_hyphen=False, 
                 predef_real_page_nums=None):
        self.path = path
        self.skip_pages = skip_pages
        self.page_num_on_top = page_num_on_top
        self.word_hyphen = word_hyphen
        self.predef_real_page_nums = predef_real_page_nums
        if os.path.islink(path):
            path = os.readlink(path)
        path_parts = os.path.split(path)
        self.doc_name = path_parts[-1]
        print(self.doc_name)
        
    def read_pages(self, verbose=False):
        result = read_hocr_pages(self.path, self.skip_pages, 
                                 self.page_num_on_top, 
                                 self.word_hyphen, 
                                 self.predef_real_page_nums, 
                                 verbose)
        self.txt_prep, self.line_start_positions, self.real_page_nums, self.line_ys = result
        self.line_start_positions_inv = {v: k for k, v in self.line_start_positions.items()}
        self.real_page_nums_inv = {v: k for k, v in self.real_page_nums.items()}
        return result

    ### CONVERTERS ###
    
    def convert_batch_positions_to_page_coords(self, positions):
        return convert_batch_positions_to_page_coords(positions, self.line_start_positions)
    
    def convert_position_to_page_coords(self, position):
        return convert_position_to_page_coords(position, self.line_start_positions)

    def convert_page_coords_to_position(self, page_coords):
        return convert_page_coords_to_position(page_coords, self.line_start_positions_inv)

    def convert_real_to_page_coords(self, real_page_coords):
        return convert_real_to_page_coords(real_page_coords, self.real_page_nums_inv)

    def convert_page_to_real_coords(self, page_coords):
        return convert_page_to_real_coords(page_coords, self.real_page_nums)
    
    def convert_position_to_real_coords(self, position):
        page_coords = self.convert_position_to_page_coords(position)
        return self.convert_page_to_real_coords(page_coords)

    def convert_real_coords_to_position(self, real_page_coords):
        page_coords = self.convert_real_to_page_coords(real_page_coords)
        return self.convert_page_coords_to_position(page_coords)

