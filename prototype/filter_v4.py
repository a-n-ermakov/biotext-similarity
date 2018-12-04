from collections import defaultdict


def filter_noise(links, eps=200, n_steps=5, n_links_after=25000, verbose=False):
    '''Filtering noise using 3-points scheme upon reaching n_links_after or n_steps
    Input:
      links:list(tuple(int*2)) - list of sample position, reference position
      eps:int - area size to find next chain element
      n_steps:int - number of steps
      n_links_after:int - number of links after filtering (early stop)
      verbose:bool - print verbose messages
    Output:
      list(tuple(int*2)) - filtered links
    '''
    for step_idx in range(n_steps):
        count_before = len(links)
        if verbose:
            print(f'FILTERING positions step {step_idx+1} of {n_steps} ...')
        links_dict = defaultdict(list)
        for sample_pos, ref_pos in links:
            links_dict[sample_pos].append(ref_pos)

        sample_positions = sorted(links_dict.keys())
        links_filtered = []
        for s1, s2, s3 in zip(sample_positions[:-2], sample_positions[1:-1], sample_positions[2:]):
            if s2 - s1 > eps and s3 - s2 > eps:
                continue
            for r2 in links_dict[s2]:
                r1s = [r1 for r1 in links_dict[s1] if r1 < r2]
                r3s = [r3 for r3 in links_dict[s3] if r2 < r3]            # or
                if s2 - s1 <= eps and any([r2 - r1 <= eps for r1 in r1s]) and \
                   s3 - s2 <= eps and any([r3 - r2 <= eps for r3 in r3s]):
                    links_filtered.append((s2, r2))
        count_after = len(links_filtered)
        if verbose:
            print(f'FILTERING before|after|filtered {count_before}|{count_after}|{count_before-count_after}')
        links=links_filtered
        if len(links) < n_links_after:
            if verbose:
                print(f'FILTERING early stop by reaching {n_links_after}')
            break
    
    return links


