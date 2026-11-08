"""GT-free missing-observation restoration, not proof of existence or identity."""


def restore(groups,pool,minimum_confidence=.05):
    if not 0<=minimum_confidence<=1:
        raise ValueError('Invalid residual confidence')
    result = [list(g) for g in groups]
    represented = set()
    for group in result:
        if not group or len(group)!=len(set(group)):
            raise ValueError('Invalid seed group')
        if represented.intersection(group):
            raise ValueError('Seed repeats an observation across objects')
        if any(u not in pool for u in group):
            raise ValueError('Missing source observation')
        represented.update(group)
    added = [u for u,d in sorted(pool.items()) if u not in represented and d['conf']>=minimum_confidence]
    # Singletons are unverified predictions, not GT-aware replacement choices.
    result.extend([u] for u in added)
    return result,added
