"""Offline token-slot constrained batching; this is not a CUDA memory allocator."""


def rounded(length, multiple=8):
    if length < 1 or multiple < 1:
        raise ValueError('positive length and multiple required')
    return ((length+multiple-1)//multiple)*multiple


def distinct_plans(plans):
    """Keep one measurement per identical ordered batch plan, with explicit aliases."""
    unique={}; aliases={}
    for name,plan in plans.items():
        same=next((other for other,existing in unique.items() if existing==plan),None)
        if same is None: unique[name]=plan
        else: aliases[name]=same
    return unique,aliases


def token_budget_batches(lengths, token_budget, output_reserve, max_sequences=8, pad_multiple=8):
    """Length-sorted greedy packing with B*(rounded(max input)+output cap) bound.

    Reserves padded KV sequence slots through the full generation cap. Model
    weights, activations and attention workspace are not represented by slots.
    """
    if token_budget < 1 or output_reserve < 0 or max_sequences < 1 or pad_multiple < 1:
        raise ValueError('invalid scheduling limits')
    if any(not isinstance(n,int) or n<1 for n in lengths):
        raise ValueError('lengths must be positive integers')
    ordered=sorted(range(len(lengths)),key=lambda i:lengths[i])
    result=[]; batch=[]
    for i in ordered:
        if rounded(lengths[i],pad_multiple)+output_reserve>token_budget:
            raise ValueError('one document cannot fit token budget; no truncation applied')
        proposed=batch+[i]
        slots=len(proposed)*(rounded(lengths[i],pad_multiple)+output_reserve)
        if batch and (len(proposed)>max_sequences or slots>token_budget):
            result.append(batch);batch=[i]
        else:
            batch=proposed
    if batch: result.append(batch)
    return result


def plan_stats(lengths,plan,output_reserve,pad_multiple=8):
    if sorted(i for b in plan for i in b)!=list(range(len(lengths))) or any(not b for b in plan):
        raise ValueError('plan must cover each input exactly once')
    useful=sum(lengths)
    allocated=sum(len(b)*rounded(max(lengths[i] for i in b),pad_multiple) for b in plan)
    return {'batches':len(plan),'batch_sizes':[len(b) for b in plan],
            'useful_input_tokens':useful,'padded_input_positions':allocated,
            'padding_fraction':1-useful/allocated if allocated else 0,
            'max_reserved_token_slots':max((len(b)*(rounded(max(lengths[i] for i in b),pad_multiple)+output_reserve) for b in plan),default=0)}
