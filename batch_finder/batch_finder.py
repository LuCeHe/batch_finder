"""
Batch Finder - Core functionality for finding maximum batch sizes, docs, and timesteps.
"""

import os
import time
import threading
import traceback
from typing import Optional, Callable, Dict, Any, Tuple
import torch
from tqdm import tqdm


def check_memory_max(
        batch_size=2,
        gdmc_args=None,
        n_docs=4,
):
    _, model, _, _ = get_data_model_and_collator(**gdmc_args)

    input_ids = torch.randint(0, 1, (batch_size, max_seq_length))
    attention_mask = torch.ones((batch_size, max_seq_length))
    input_ids_encoder = torch.randint(0, 1, (n_docs, batch_size, max_seq_length))
    attention_mask_encoder = torch.ones((n_docs, batch_size, max_seq_length))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    input_ids_encoder = input_ids_encoder.to(device)
    attention_mask_encoder = attention_mask_encoder.to(device)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_ids_encoder=input_ids_encoder,
        attention_mask_encoder=attention_mask_encoder,
    )

    loss = output.logits.sum()
    loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def find_max_batch(
        forward_and_backward: Callable,
        delay: float = 5.0,
        max_batch_size: int = 1024,
        n_attempts: int = 50,
        n_positive_stops: int = 3,
):
    """
    Find the maximum batch size that can be processed without running out of memory.
    Args:
        model_and_loss_fn (Callable): A function that returns the model and loss function.




        delay (float): Delay in seconds between attempts to allow memory to clear.
        max_batch_size (int): The initial maximum batch size to test.
        n_attempts (int): The maximum number of attempts to find the maximum batch size.
    """

    print("\nStarting search for maximal batch_size...\n")

    pbar = tqdm(range(n_attempts), total=100, desc="Overall Progress", position=0)

    successful = []
    unsuccessful = []
    for i in pbar:
        print(f"{i+1}/{n_attempts} Attempting to find max batch size... successful={successful}, unsuccessful={unsuccessful}")

        if len(successful) == 0 and len(unsuccessful) == 0:
            batch_size_i = max_batch_size
        else:
            last_max_good = max(successful) if successful else max_batch_size
            last_min_bad = min(unsuccessful) if unsuccessful else 1
            batch_size_i = (last_max_good + last_min_bad) // 2

        def run_test(n):
            try:
                # print(f"\n--- Testing n_docs={n} ---")
                forward_and_backward(batch_size=n)
                successful.append(n)
                # print(f"✓ Finished n_docs={n}")
                pbar.set_postfix({'batch_size': n, 'status': '✅', 'current_max': max(successful)})

            except Exception as e:
                # print(f"⚠️ Error during n_docs={n}: {e}")
                # traceback.print_exc()
                print(e)
                unsuccessful.append(n)
                pbar.set_postfix(
                    {'batch_size': n, 'status': '❌', 'current_max': max(successful) if successful else 'N/A'})

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(delay)

        t = threading.Thread(target=run_test, args=(batch_size_i,))
        t.start()
        t.join()

        if len(successful) >= n_positive_stops:
            print(f"\nStopping tests after {n_positive_stops} successes at batch_size={successful}.")
            break

    if successful:
        max_batch_found = max(successful)
        print(f"\n✅ Max batch_size that passed without error: {max_batch_found}")
    else:
        print("\n❌ No batch_size value passed without error.")
        max_batch_found = None

    return max_batch_found
