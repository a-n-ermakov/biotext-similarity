import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import DBSCAN
from page_reader import convert_batch_positions_to_page_coords
from filter_v4 import filter_noise


def sequencing(genome, ls_positions, sample_size=(10, 10), sample_count=0.2, verbose=False):
    '''Sequencing of genome to samples for futher alignment
    Input:
      genome:str - genome
      ls_positions:dict(int,tuple(int,int)) - dict of line start position->page coords(page idx, line on page idx) in genome
      sample_size:tuple(int*2) - inclusive size [max, min] of sample (fragment)
      sample_count:float - count of samples as rate from genome length
      verbose:bool - print verbose messages
    Output:  
      dict(int,str) - dict of position->sample
      dict(int,tuple(int*2)) - dict of sample position->page coords(page idx, line on page idx)
    '''
    if verbose:
        print(f'SEQUENCING samples...')
    samples_count = int(sample_count * len(genome))
    low, high = sample_size
    pos_high = len(genome) - max(low, high)
    
    positions = np.sort(np.random.randint(pos_high, size=samples_count))
    sizes = np.random.randint(low, high+1, samples_count)

    samples = {}
    for pos, size in zip(positions, sizes):
        samples[pos] = genome[pos:pos+size]

    page_coords = convert_batch_positions_to_page_coords(positions, ls_positions)
    
    if verbose:
        total_count = len(samples)
        unique_count = len(set(samples.values()))
        dup_count = total_count-unique_count
        print(f'SEQUENCING: sample_size {sample_size}, sample_count {sample_count}')
        print(f'SEQUENCING: samples generated total|unique|duplicated {total_count}|{unique_count}|{dup_count}')
        print(f'SEQUENCING: page_coords {len(page_coords)}')
    
    return samples, page_coords


def alignment(ref_genome, ref_ls_positions, samples, verbose=False):
    '''Alignment of genome samples to reference genome
    Input:
      ref_genome:str - reference genome
      ref_ls_positions:dict(int,tuple(int*2)) - dict of line start position->page coords(page idx, line on page idx) in reference genome
      samples:dict(int,str) - dict of genome samples
      verbose:bool - print verbose messages
    Output:
      list(tuple(int*2)) - list of sample position, reference position
      dict(int,tuple(int*2)) - dict of reference position->page coords(page idx, line on page idx)
    '''
    if verbose:
        print(f'ALIGNMENT samples...')
    sample_links = defaultdict(list)
    for sample in set(samples.values()):
        cur_ref_positions = [m.start() for m in re.finditer(sample, ref_genome)]
        if cur_ref_positions:
            sample_links[sample] = cur_ref_positions
    unique_count = len(sample_links)
 
    links = [(s, r) for s, sample in samples.items() for r in sample_links[sample]]
    total_count = len(links)
    
    ref_positions = sorted(set([x for _, x in links]))
    ref_page_coords = convert_batch_positions_to_page_coords(ref_positions, ref_ls_positions)
    
    if verbose:
        dup_count = total_count-unique_count
        print(f'ALIGNMENT: samples located total|unique|duplicated {total_count}|{unique_count}|{dup_count}')
        print(f'ALIGNMENT: ref_page_coords {len(ref_page_coords)}')
    
    return links, ref_page_coords


def group_links(links, by_sample=True):
    '''Grouping links by sample pos
    Input:
      links:list(tuple(int*2)) - list of sample position, reference position
      by_sample:bool - if True group by sample otherwise group by ref
    Output:
      dict(int,list(int)) - dict of position->list of positions
    '''
    groups = defaultdict(list)
    for sample_pos, ref_pos in links:
        k, v = (sample_pos, ref_pos) if by_sample else (ref_pos, sample_pos)
        groups[k].append(v)
    return groups


def enrich_page_coords(df, ref_page_coords, sample_page_coords, verbose=False):
    '''Enrich cluster dataframe with page coords
    Input:
      df:pd.DataFrame - dataframe with `label`, `ref_pos` and `sample_pos` columns
      ref_page_coords, sample_page_coords:dict(int,tuple(int*2)) - dict of reference and sample position->page coords (page idx, line on page idx)
      verbose:bool - print verbose messages
    Output:
      pd.DataFrame - dataframe with each cluster limits in positions and page coords for ref and sample
    '''
    areas = []
    for label, dfl in df.groupby('label'):
        ref_pos, sample_pos = dfl['ref_pos'], dfl['sample_pos']
        
        ref_begin, ref_end = ref_pos.min(), ref_pos.max()
        ref_begin_page_coords = ref_page_coords[ref_begin]
        ref_end_page_coords = ref_page_coords[ref_end]
        
        sample_begin, sample_end = sample_pos.min(), sample_pos.max()
        sample_begin_page_coords = sample_page_coords[sample_begin]
        sample_end_page_coords = sample_page_coords[sample_end]
        
        if verbose:
            print('COORDS: cluster # %3d, sample [%6d .. %6d], ref [%6d .. %6d]' % 
                  (label, sample_begin, sample_end, ref_begin, ref_end))
        
        #todo use coords for end character of sample
        cur_area = ref_begin, ref_begin_page_coords, ref_end, ref_end_page_coords, \
                   sample_begin, sample_begin_page_coords, sample_end, sample_end_page_coords
        areas.append(cur_area)
        
    
    columns = 'ref_begin', 'ref_begin_page_coords', 'ref_end', 'ref_end_page_coords', \
              'sample_begin', 'sample_begin_page_coords', 'sample_end', 'sample_end_page_coords'
  
    return pd.DataFrame.from_records(areas, columns=columns)


def calculate_similar_areas(links, ref_page_coords, sample_page_coords,
                            use_custom_metric=False, custom_metric_max_samples=25000, tau=0.1,
                            eps=500, min_samples=75, show=False, fn_clusters=None, verbose=False):
    '''Calculating similar areas in target and reference genomes
    Input:
      links:list(tuple(int*2)) - list of sample position, reference position
      ref_page_coords, sample_page_coords:dict(int,tuple(int*2)) - dict of reference and sample position->page coords (page idx, line on page idx)
      use_custom_metric:bool - usage of custom metric for clustering (euclidean otherwise)
      custom_metric_max_samples:int - max number of samples for custom metric, using euclidean when exceeded
      tau:float - custom metric param
      eps, min_samples:int - clustering params
      show:bool - show scatter of clusters
      fn_clusters:str - filename for save result cluster diagram
      verbose:bool - print verbose messages
    Output:
      pd.DataFrame - areas dataframe
    '''
    df = pd.DataFrame.from_records(links, columns=['sample_pos', 'ref_pos'])
    
    if use_custom_metric and df.shape[0] < custom_metric_max_samples:
        import build.distance_matrix as dmc
        from scipy.sparse import coo_matrix
        if verbose:
            print(f'SIMILARITY: distance matrix for {df.shape[0]} samples, tau={tau}')
        coo_arr = dmc.distance_matrix_45(df.values, tau=tau)
        distance_matrix = coo_matrix((coo_arr[:,0], (coo_arr[:,1].astype('int32'), coo_arr[:,2].astype('int32'))))       
        if verbose:
            print(f'SIMILARITY: distance matrix have {coo_arr.shape[0]} values with {distance_matrix.shape} shape')
            print(f'SIMILARITY: custom metric, eps={eps}, min_samples={min_samples}')
        dbscan = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples, n_jobs=-1)
        dbscan.fit(distance_matrix)
    else:
        if verbose:
            print(f'SIMILARITY: euclidean metric, eps={eps}, min_samples={min_samples}')
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        dbscan.fit(df)
    
    df['label'] = dbscan.labels_
    signal_mask = dbscan.labels_ != -1
    dfc = df[signal_mask]

    if show:
        print('SIMILARITY: clusters count:', dfc['label'].nunique())
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.scatter(dfc['ref_pos'], dfc['sample_pos'], s=1, c=dfc['label'], alpha=0.5, cmap='viridis')
        plt.xlabel('ref_pos')
        plt.ylabel('sample_pos')
        plt.colorbar()
        plt.grid()
        plt.gcf().set_size_inches(10, 8)
        if fn_clusters is not None:
            fig.set_size_inches(12, 10)
            fig.savefig(fn_clusters)
    
    dfc = enrich_page_coords(dfc, ref_page_coords, sample_page_coords, verbose)
    return dfc


class SingleBioModel():
    '''Model wrapper based on bio engine for one sample and one reference genome'''
    
    def __init__(self, sample_reader, ref_reader, sample_size=(10,10), sample_count=0.2):
        self.sample_reader = sample_reader
        self.ref_reader = ref_reader
        self.sample_size = sample_size
        self.sample_count = sample_count

    def fit(self, verbose=False):
        self.samples, self.sample_page_coords = sequencing(self.sample_reader.txt_prep, 
                                                           self.sample_reader.line_start_positions, 
                                                           self.sample_size,
                                                           self.sample_count, 
                                                           verbose=verbose)

    def transform(self, eps=100, n_steps=5, n_links_after=25000, verbose=False):
        self.links_raw, self.ref_page_coords = alignment(self.ref_reader.txt_prep, 
                                                         self.ref_reader.line_start_positions, 
                                                         self.samples, 
                                                         verbose)
        self.links = filter_noise(self.links_raw,
                                  eps=eps,
                                  n_steps=n_steps,
                                  n_links_after=n_links_after,
                                  verbose=verbose)
        
    def fit_transform(self, eps=100, n_steps=5, n_links_after=25000, verbose=False):
        self.fit(verbose=verbose)
        self.transform(eps=eps, 
                       n_steps=n_steps, 
                       n_links_after=n_links_after, 
                       verbose=verbose)

    def predict(self, use_custom_metric=False, custom_metric_max_samples=25000, tau=0.1,
                eps=500, min_samples=50, show=False, fn_clusters=None, verbose=False):
        self.areas_df = calculate_similar_areas(self.links,
                                                self.ref_page_coords,
                                                self.sample_page_coords,
                                                use_custom_metric=use_custom_metric,
                                                custom_metric_max_samples=custom_metric_max_samples,
                                                tau=tau,
                                                eps=eps,
                                                min_samples=min_samples,
                                                show=show,
                                                fn_clusters=fn_clusters,
                                                verbose=verbose)
