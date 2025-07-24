import phfork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import time


def process_pka_batch(args):
    """Worker function to process a batch of pKa values"""
    pka_batch, known_values, citrate_molarity, counter, lock = args
    batch_results = []
    
    for pka in pka_batch:
        citricacid = phfork.AcidAq(pKa=pka, charge=0, conc=citrate_molarity)
        ratios = []
        error = 0.0
        
        for i in range(0, 201):
            na_molarity = citrate_molarity * 3 * (i / 100 / 2)
            na = phfork.IonAq(charge=1, conc=na_molarity)
            system = phfork.System(citricacid, na)
            system.pHsolve()

            ratios.append(
                {
                    'pH': round(system.pH, 2),
                    'acid ratio': 100 - i,
                    'base ratio': i,
                }
            )

        for ph in known_values:
            closest_ph = min(ratios, key=lambda d: abs(d['pH'] - ph['ph']))
            error += (ph['acid ratio'] - closest_ph['acid ratio']) ** 2

        batch_results.append(
            {
                'error': error,
                'pkas': pka,
            }
        )
        
        # Update progress counter
        with lock:
            counter.value += 1
    
    return batch_results


def main():
    citrate_molarity = 0.1

    imported = pd.read_csv('bufferdata.csv')
    imported = imported.to_dict(orient='split', index=False)

    known_values = []

    for data in imported['data']:
        known_values.append(
            {'ph': data[0], 'acid ratio': data[1], 'base ratio': data[2]}
        )

    pkas = []

    for pka1 in range(200, 350):
        for pka2 in range(400, 550):
            for pka3 in range(550, 650):
                pkas.append([pka1 / 100, pka2 / 100, pka3 / 100])

    print(f'This many trials: {len(pkas)}')

    # Split pkas into batches for multiprocessing
    num_processes = mp.cpu_count()
    batch_size = len(pkas) // num_processes
    pka_batches = [pkas[i:i + batch_size] for i in range(0, len(pkas), batch_size)]
    
    # Create shared counter and lock for progress tracking
    manager = mp.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    # Prepare arguments for worker processes
    worker_args = [(batch, known_values, citrate_molarity, counter, lock) for batch in pka_batches]
    
    # Use multiprocessing to process batches in parallel with trial progress
    print(f"Processing {len(pkas)} trials across {num_processes} processes...")
    
    with tqdm(total=len(pkas), desc="Processing trials") as pbar:
        with mp.Pool(processes=num_processes) as pool:
            # Start async job
            result = pool.map_async(process_pka_batch, worker_args)
            
            # Monitor progress
            while not result.ready():
                current_count = counter.value
                pbar.n = current_count
                pbar.refresh()
                time.sleep(0.1)
            
            # Get final results
            batch_results = result.get()
            pbar.n = len(pkas)
            pbar.refresh()
    
    # Flatten results from all batches
    out = []
    for batch_result in batch_results:
        out.extend(batch_result)

    pka = min(out, key=lambda x: x['error'])

    with open('pka.txt', 'w') as f:
        f.write(pka)

    print(pka)

    ratios = []
    citricacid = phfork.AcidAq(pKa=pka['pkas'], charge=0, conc=citrate_molarity)
    for i in range(0, 101):
        na_molarity = citrate_molarity * 3 * (i / 100)
        na = phfork.IonAq(charge=1, conc=na_molarity)
        system = phfork.System(citricacid, na)
        system.pHsolve()

        ratios.append(
            {
                'pH': round(system.pH, 2),
                'acid ratio': 100 - i,
                'base ratio': i,
            }
        )

    out = pd.DataFrame.from_dict(ratios)
    out.to_csv('ratios.csv', index=False)

    phs = np.linspace(1, 9, 1000)

    fracs = citricacid.alpha(phs)

    plt.plot(phs, fracs)

    plt.legend(['H3A', 'H2A-', 'HA2-', 'A3-'])

    plt.savefig('graph.png')


if __name__ == '__main__':
    main()
