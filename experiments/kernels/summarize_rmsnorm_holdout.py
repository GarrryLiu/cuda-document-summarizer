"""Render measured holdout results without claiming equal-work or factuality."""
import argparse
import json
from pathlib import Path


def render(report):
    if report['status'] != 'complete':
        raise ValueError('Refuse to summarize an incomplete study as completed')
    rows = report['results']
    lines = ['## Measured held-out results', '',
             f"GPU: {report['gpu']}. {len(rows)} intact contracts, batch one, "
             f"{report['repeats']} randomized rounds, cap {report['max_new_tokens']} tokens.", '',
             'Times below are per-document medians. Positive reductions mean faster.', '',
             '| Prompt tokens | Native total s | Compiled total s | CUDA total s | CUDA total reduction vs native | CUDA prefill reduction vs native | CUDA/native identical rounds |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m = row['median']; n, c = m['native'], m['cuda']
        lines.append(f"| {row['prompt_tokens']} | {n['seconds']:.3f} | {m['compiled']['seconds']:.3f} | {c['seconds']:.3f} | "
                     f"{100*(1-c['seconds']/n['seconds']):.2f}% | {100*(1-c['prefill_gpu_ms']/n['prefill_gpu_ms']):.2f}% | "
                     f"{sum(row['paired_token_equal']['cuda'])}/{report['repeats']} |")
    lines += ['', '### Output and workload checks', '']
    for mode in ('native', 'compiled', 'cuda'):
        runs = [run for row in rows for run in row['runs'][mode]]
        stable = sum(len({r['output_sha256'] for r in row['runs'][mode]}) == 1 for row in rows)
        lines.append(f"- {mode}: {sum(r['ended_with_eos'] for r in runs)}/{len(runs)} generations ended with EOS; "
                     f"{sum(r['cap_hit'] for r in runs)} hit the cap; {stable}/{len(rows)} documents had identical tokens across repeated runs.")
    for mode in ('compiled', 'cuda'):
        paired = [v for row in rows for v in row['paired_token_equal'][mode]]
        total_n = sum(row['median']['native']['seconds'] for row in rows)
        total_m = sum(row['median'][mode]['seconds'] for row in rows)
        lines.append(f"- {mode} vs native: {sum(paired)}/{len(paired)} token-identical pairs; "
                     f"sum of per-document median generation times changes by {100*(total_m/total_n-1):+.2f}%.")
    lines += ['', 'Output-length differences make aggregate wall-time changes unequal-work comparisons. '
              'Prefill uses the same source tokens; token throughput is recorded separately. '
              'EOS, token identity and timing do not certify legal factuality.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('output'); a = p.parse_args()
    Path(a.output).write_text(render(json.loads(Path(a.input).read_text())))
