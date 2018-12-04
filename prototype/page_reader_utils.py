import re

RE_TEXT = re.compile('([^а-яёa-z0-9\n\r])+', re.IGNORECASE)
RE_MULTI_SPACE = re.compile('[ ]+')
RE_MULTI_EOL = re.compile('([\n\r])+')
RE_PAGE_NUM = re.compile('(\d)+')


def clean_txt(s, multi_eol=True):
    '''Clean page text from all but letters, numbers, hyphens and EOLs
    In:
      s:str - text
      multi_eol:bool - remove multi EOL characters
    Out:
      str - preprocessed text
    '''
    txt = s.lower()
    txt = RE_TEXT.sub(' ', txt)
    txt = RE_MULTI_SPACE.sub(' ', txt)
    if multi_eol:
        txt = RE_MULTI_EOL.sub('\n', txt)
    return txt
      

def find_occurences(s, ch, shift=0):
    '''Find all character occurences in string
    Input:
      s:str - string
      ch:str - character
      shift:int - shift for addition to all indexes
    Output:
      list(int) - indexes in str
    '''
    return [i+shift for i, letter in enumerate(s) if letter == ch]


def extract_real_page_num(p, line, verbose=False):
    '''Extract real page number using OCR and previous page numbers
       Function modifies `real_page_nums` dict
    Input:
      p:int - page file index
      first_line:str - line with number
      verbose:bool - print verbose messages
    Output:
      int - real page number (None if not extracted)
    '''
    if len(line) < 5:
        m = RE_PAGE_NUM.search(line)
        real_page_num = None if m is None else int(m.group(0))
        if verbose:
            print(f'[{p}|{real_page_num}] ', end='')
    else:
        real_page_num = None
        if verbose:
            print(f'[{p}|{line[:10]}] ', end='')
    
    return real_page_num


# deprecated - use new version below
def fix_real_page_nums_old(real_page_nums):
    '''Fix real page numbers: remove noise and OCR misses
    Input:
      real_page_nums:dict(int,int) - parsed real page numbers
    Output:
      dict(int,int) - fixed real page numbers
    '''
    page_indices = sorted(real_page_nums.keys())
    idx_page_nums, fixed_real_page_nums = {}, {}
    for i, p in enumerate(page_indices):
        real_page_num = real_page_nums[p]
        if real_page_num is not None and real_page_num > 0:
            if i > 0 and real_page_num in fixed_real_page_nums:
                # fix incorrect
                real_page_num = idx_page_nums[i-1] + 1
        else:
            if i > 0:
                # fix incorrect
                real_page_num = idx_page_nums[i-1] + 1
            else:
                real_page_num = 1

        fixed_real_page_nums[p] = real_page_num
        idx_page_nums[i] = real_page_num
        
        # fix single noise numbers
        if i > 1 and idx_page_nums[i-2] == real_page_num - 2 and idx_page_nums[i-1] != real_page_num - 1:
            fixed_real_page_nums[page_indices[i-1]] = real_page_num - 1
            idx_page_nums[i-1] = real_page_num - 1
    
    return fixed_real_page_nums


def fix_real_page_nums(real_page_nums, verbose=False):
    '''Fix real page numbers: remove noise and OCR misses - new chains algorythm
    Input:
      real_page_nums:dict(int,int) - parsed real page numbers
      verbose:bool - print verbose messages
    Output:
      dict(int,int) - fixed real page numbers
    '''
    if verbose:
        print('REAL_PAGE_NUM_FIX: total pages', len(real_page_nums))
    # 1.searching chains
    chains = []
    page_indices = range(len(real_page_nums))
    for page_idx in page_indices:
        real_page_num = real_page_nums[page_idx]
        if real_page_num is not None:
            chain_found = False
            for chain in chains:
                last_page_idx, last_real_page_num = chain[-1]
                if page_idx - last_page_idx == real_page_num - last_real_page_num:
                    # exist chain
                    chain_found = True
                    chain.append((page_idx, real_page_num))
            if not chain_found:
                # new chain
                chain = [(page_idx, real_page_num)]
                chains.append(chain)
    if verbose:
        print('REAL_PAGE_NUM_FIX: chains found', len(chains))
    # 2.selecting biggest chain for each page as reper point
    page_repers = []
    for page_idx in page_indices:
        chain_selected = None
        for chain in chains:
            start_page_idx = chain[0][0]
            finish_page_idx = chain[-1][0]
            if start_page_idx <= page_idx <= finish_page_idx and \
              (chain_selected is None or len(chain) > len(chain_selected)):
                chain_selected = chain
        page_repers.append(chain_selected[0] if chain_selected is not None else None)
    if verbose:
        pages_found_count = len([x for x in page_repers if x is not None])
        print('REAL_PAGE_NUM_FIX:', len(set(page_repers)), 'chains selected for', pages_found_count, 'pages')
    # 3.fixing page nums using repers (current and last)
    fixed_real_page_nums = {}
    page_repers_non_empty = [x for x in page_repers if x is not None]
    last_reper = page_repers_non_empty[0] if page_repers_non_empty else None
    if last_reper is not None:
        for page_idx, page_reper in zip(page_indices, page_repers):
            reper = page_reper if page_reper is not None else last_reper
            last_page_idx, last_real_page_num = reper
            delta = page_idx - last_page_idx
            fixed_real_page_nums[page_idx] = last_real_page_num + delta
            last_reper = reper
    if verbose:
        print('REAL_PAGE_NUM_FIX: fixed real page nums', fixed_real_page_nums)
    
    return fixed_real_page_nums
