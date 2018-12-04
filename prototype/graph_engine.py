from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os


def draw_link(img, ref_box, sample_box, highlights=[], highlights_segments=10,
              result_fn=None, line_width=5, line_color=(255, 0, 0, 255)):
    '''Draw the link between ref and sampl boxes: 
       from left side of ref to right side of sample
    Input:
      img:pillow.Image - coloration
      ref_box, sample_box:tuple(int*4) - boxes with (left, upper, right, lower) coordinates
      highlights: list(str) - some of highlight flags : 'ref_top', 'ref_bottom', 'sample_top', 'sample_bottom'
      highlights_segments: int - number of segments of dashed line
      result_fn:str - filename for result file save
      line_width:int - line width (5 by default)
      line_color:tuple(int*4) - line color (red by default)
    Output:
      PIL.Image - coloration image
    '''
    img = img.copy()
    draw = ImageDraw.Draw(img)
    
    # boxes
    ref_left, ref_upper, ref_right, ref_lower = ref_box
    sample_left, sample_upper, sample_right, sample_lower = sample_box
    for i in range(line_width):
        draw.rectangle((ref_left+i, ref_upper+i, ref_right-i, ref_lower-i), None, line_color)
        draw.rectangle((sample_left+i, sample_upper+i, sample_right-i, sample_lower-i), None, line_color)

    # highlights over the boxes for dashed sides (top and bottom of page)
    for h in highlights:
        if h in ('ref_top', 'ref_bottom'):
            h_left, h_right = ref_left, ref_right
        else:
            h_left, h_right = sample_left, sample_right
        dh_y = line_width // 2
        if h == 'ref_top':
            h_y = ref_upper + dh_y
        elif h == 'sample_top':    
            h_y = sample_upper + dh_y
        elif h == 'ref_bottom':
            h_y = ref_lower - dh_y
        elif h == 'sample_bottom':    
            h_y = sample_lower - dh_y
        else:
            continue
        
        segment_width = (h_right-h_left) // (highlights_segments*2 + 1)
        bg_color = (255, 255, 255, 255) # white
        for j in range(1, highlights_segments+1):
            segment_right = h_left + 2*j*segment_width
            segment_left = segment_right - segment_width
            begin = segment_left, h_y
            end = segment_right, h_y
            draw.line(begin + end, bg_color, line_width + 2)

    # link
    begin = ref_left, np.mean([ref_upper, ref_lower])
    end = sample_right, np.mean([sample_upper, sample_lower])
    draw.line(begin + end, line_color, line_width)
    
    # arrow
    x, y = begin[0] - end[0], begin[1] - end[1]
    arrow_len = abs(x) // 5
    arrow_angle = np.pi / 6
    angle = np.arctan2(y, x)
    up_angle, down_angle = angle + arrow_angle / 2, angle - arrow_angle / 2
    up_arrow_end = end[0] + arrow_len * np.cos(up_angle), end[1] + arrow_len * np.sin(up_angle)
    down_arrow_end = end[0] + arrow_len * np.cos(down_angle), end[1] + arrow_len * np.sin(down_angle)
    draw.polygon(end + up_arrow_end + down_arrow_end, line_color, line_color)
    
    del draw
    if result_fn is not None:
        img.save(result_fn)
    return img


def convert_to_box(img, crop_box, line_range, line_ys, margin=50, show=False, verbose=False):
    '''Convert line range to line box
    Input:
      img:PIL.Image - coloration
      crop_box:tuple(int*4) - crop box to extract image from coloration
      line_range:tuple(int*2) - range (start, finish) with line indexes, if index is None - begin|end of page
      line_ys:list:tuple(int*2) - y coordinates of upper and lower edges of lines
      margin:int - margins from left and rigth sides of page
      show:bool - show graphs
      verbose:bool - print verbose messages
    Output:
      tuple(int*4) - coordinates on image, if coord is None - begin|end of page
    '''
    ys = line_ys
    start, finish = line_range
    start = start if start is not None else 0
    finish = finish if finish is not None else len(ys)-1
    if not 0 <= start <= finish < len(ys):
        if verbose:
            print(f'incorrect line range: start {start}, finish {finish}, detected {len(ys)}')
        return None
    if start == 0 or finish+1 == len(ys):
        h = int(np.mean(np.array(ys[1]) - np.array(ys[0]))) // 2
    left, edge_border, right, _ = crop_box
    upper = ys[start][0] - h if start == 0 else np.mean([ys[start][0], ys[start-1][1]])
    lower = ys[finish][1] + h if finish+1 == len(ys) else np.mean([ys[finish][1], ys[finish+1][0]])
    return left + margin, upper + edge_border, right - margin, lower + edge_border


def convert_line_ys(line_ys, img, thumb_width, thumb_height, verbose=False):
    '''Convert line ys for thumbnails, which is transformed to thumb size
    Input:
      line_ys:list(tuple(int*2)) - y coordinates of upper and lower edges of lines
      img:PIL.Image - image and his opponent
      thumb_width,thumb_height:int - thumbnail width and height
      verbose:bool - print verbose messages
    Output:
      list(tuple(int*2)) - transformed y coordinates of upper and lower edges of lines
    '''
    k, b = 1.0, 0.0
    k_height = float(thumb_height) / img.height
    k_width = float(thumb_width) / img.width
    if k_height < 1.0 or k_width < 1.0:
        k = min(k_height, k_width)
        b = int(thumb_height - k*img.height)
        if verbose:
            print(f'Converting line_ys using k={k} and b={b}')
        return [(k*x + b, k*y + b) for x, y in line_ys]
    else:
        return line_ys


def prepare_coloration(ref_img_fn, sample_img_fn, ref_range, sample_range, 
                       ref_doc_name, sample_doc_name,
                       ref_line_ys, sample_line_ys,
                       result_fn=None, thumb_width=1050, thumb_height=1485,
                       edge_border=60, middle_border=200, 
                       line_width=5, font_size=20,
                       show=False, verbose=False):
    '''Prepare image coloration: ref image at right, sample image at left, and arrow between them
    Input:
      ref_img_fn, sample_img_fn:str - filenames of ref and sample page_images
      ref_range, sample_range:tuple(int*2) - ranges (start, finish) with line indexes, if index is None - begin|end of page
      ref_doc_name, sample_doc_name:str - names of ref and sample documents
      ref_line_ys,sample_line_ys:list(tuple(int*2)) - y coordinates of upper and lower edges of ref and sample lines
      result_fn:str - filename for result file save
      thumb_width, thumb_height:int - thumb size
      edge_border, middle_border:int - border sizes on edge and between images
      line_width:int - line width
      font_size:int - for texts
      show:bool - show graphs
      verbose:bool - print verbose messages
    Output:
      PIL.Image - coloration image or None if error occured
    '''
    
    # creating final image and pasting ref and sample images
    ref_img = Image.open(ref_img_fn)
    ref_img.load()
    sample_img = Image.open(sample_img_fn)
    sample_img.load()

    thumb_width = min(thumb_width, ref_img.width, sample_img.width)
    thumb_height = min(thumb_height, ref_img.height, sample_img.height)
    ref_line_ys = convert_line_ys(ref_line_ys, ref_img, thumb_width, thumb_height, verbose=verbose)
    sample_line_ys = convert_line_ys(sample_line_ys, sample_img, thumb_width, thumb_height, verbose=verbose)
    if verbose:
        print(f'Creating ref thumbnail with (w,h)=({thumb_width},{thumb_height}) from ({ref_img.width},{ref_img.height})')
        print(f'Creating sample thumbnail with (w,h)=({thumb_width},{thumb_height}) from ({sample_img.width},{sample_img.height})')
    ref_img.thumbnail((thumb_width, thumb_height), Image.ANTIALIAS)
    sample_img.thumbnail((thumb_width, thumb_height), Image.ANTIALIAS),

    width = edge_border + thumb_width + middle_border + thumb_width + edge_border
    height = edge_border + thumb_height + edge_border
    size = width, height

    bg_color = (127, 127, 127, 255) # gray
    wf_color = (255, 255, 255, 255) # white fields for images with alpha channel 
    img = Image.new('RGB', size, bg_color)  
    draw = ImageDraw.Draw(img)

    pos_upper, pos_lower = edge_border, edge_border + thumb_height

    ref_pos_left = edge_border + thumb_width + middle_border
    draw.rectangle((ref_pos_left, pos_upper, ref_pos_left + thumb_width, pos_lower), wf_color)
    ref_mask = ref_img.split()[1] if ref_img.mode == 'LA' else None
    img.paste(ref_img, (ref_pos_left, pos_upper), ref_mask)
    ref_img.close()
    
    sample_pos_left = edge_border
    sample_mask = sample_img.split()[1] if sample_img.mode == 'LA' else None
    draw.rectangle((sample_pos_left, pos_upper, sample_pos_left + thumb_width, pos_lower), wf_color)
    img.paste(sample_img, (sample_pos_left, pos_upper), sample_mask)
    sample_img.close()
    
    # document names
    text_pos_upper = (edge_border - font_size) // 2
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    color = 255, 255, 0, 255  # yellow
    draw.text((ref_pos_left, text_pos_upper), ref_doc_name, fill=color, font=font)
    draw.text((sample_pos_left, text_pos_upper), sample_doc_name, fill=color, font=font)
    del(draw)
    
    # drawing link
    upper, lower = edge_border, edge_border + thumb_height
    
    ref_left = edge_border + thumb_width + middle_border
    ref_right = ref_left + thumb_width
    ref_box = ref_left, upper, ref_right, lower
    ref_box = convert_to_box(img, ref_box, ref_range, ref_line_ys, show=show, verbose=verbose)

    sample_left = edge_border
    sample_right = sample_left + thumb_width
    sample_box = sample_left, upper, sample_right, lower
    sample_box = convert_to_box(img, sample_box, sample_range, sample_line_ys, show=show, verbose=verbose)
    
    highlights = ['ref_top', 'ref_bottom', 'sample_top', 'sample_bottom']
    flags = ref_range + sample_range
    highlights = [x for x, y in zip(highlights, flags) if y is None]
    
    if ref_box is None or sample_box is None:
        return None
    else:
        return draw_link(img, ref_box, sample_box, highlights, 
                         result_fn=result_fn, line_width=line_width)


def prepare_colorations(df, ref_path, sample_path, 
                        ref_doc_name, sample_doc_name,
                        ref_line_ys, sample_line_ys,
                        page_prefix='page-', verbose=False):
    '''Prepare multiple image colorations
    Input:
      df:pd.DataFrame - data frame with coloration information
      ref_path, sample_path:str - paths to ref and sample folders with images
      ref_doc_name, sample_doc_name:str - names of ref and sample documents
      ref_line_ys, sample_line_ys:dict(int,list(tuple(int*2))) - dict page_idx->list of y coordinates of upper and lower edges
                                                                of lines for ref and sample
      page_prefix:str - page prefix in file name
      verbose:bool - print verbose messages
    Output:
      list(PIL.Image) - images
    '''
    images = []
    for idx, row in df.iterrows():
        ref_begin_p, ref_begin_l = row['ref_begin_page_coords']
        ref_end_p, ref_end_l = row['ref_end_page_coords']
        sample_begin_p, sample_begin_l = row['sample_begin_page_coords']
        sample_end_p, sample_end_l = row['sample_end_page_coords']
        
        ref_range = ref_begin_l, ref_end_l
        sample_range = sample_begin_l, sample_end_l
        
        if ref_begin_p == ref_end_p and sample_begin_p == sample_end_p:
            ref_p, sample_p = ref_begin_p, sample_begin_p
        else:
            ref_p, sample_p = ref_begin_p, sample_begin_p
            if ref_begin_p != ref_end_p:
                ref_range = ref_begin_l, None
                if verbose:
                    print(f'link#{idx:03} - showing only first pages from different ref pages: {(ref_begin_p, ref_end_p)}')
            if sample_begin_p != sample_end_p:
                sample_range = sample_begin_l, None
                if verbose:
                    print(f'link#{idx:03} - showing only first pages from different sample pages: {(sample_begin_p, sample_end_p)}')

        if verbose:
            print(f'link#{idx:03} - ref page coords: {ref_p} {ref_range} -> sample page coords: {sample_p} {sample_range}')

        ref_fn = os.path.join(ref_path, f'{page_prefix}{ref_p:03}.png')
        sample_fn = os.path.join(sample_path, f'{page_prefix}{sample_p:03}.png')

        image = prepare_coloration(ref_fn, sample_fn, ref_range, sample_range, 
                                   ref_doc_name, sample_doc_name, 
                                   ref_line_ys[ref_p], sample_line_ys[sample_p],
                                   verbose=verbose)
        if image is not None:
            images.append(image)

    return images

        
def save_to_pdf(images, fn='coloration'):
    '''Save list of images to pdf file
    Input:
      images:list(PIL.Image) - images to save
      fn:str - filename of PDF
    '''
    fn = fn + '.pdf'
    if len(images) == 0:
        print('No images for PDF - creating empty PDF')
        with open(fn, 'a'): 
            os.utime(fn, None)
    elif len(images) == 1:
        images[0].save(fn, save_all=False)
    else:    
        images[0].save(fn, save_all=True, append_images=images[1:])
    for img in images:
        img.close()


class GraphEngine:
    
    def __init__(self, sample_reader, ref_reader, sample_path, ref_path):
        self.sample_reader = sample_reader
        self.ref_reader = ref_reader
        self.sample_path = sample_path
        self.ref_path = ref_path
        
    def prepare_colorations(self, df, fn='coloration', page_prefix='page-', verbose=False):
        images = prepare_colorations(df, 
                                     self.ref_path, self.sample_path,
                                     self.ref_reader.doc_name, self.sample_reader.doc_name,
                                     self.ref_reader.line_ys, self.sample_reader.line_ys,
                                     page_prefix=page_prefix, 
                                     verbose=verbose)
        save_to_pdf(images, fn)
                                     
        
