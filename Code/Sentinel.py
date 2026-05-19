# Ideally we want the code to be robust: it shouldn't just die for no reason and then give up

# we also need it to be able to resume off of the database without losing any data: all "thinking" should be stored
# outside the memory of this algo

# We need to have the concept of LOW RANK INTEGRATION which will be able to handle both the database plus injecting
# matrices into a matrix (Recursion)

# Loss
# Token Accuracy
# Cosine Similarity?
# Perplexity
# KL Divergence

# ELBo (Evidence Lower Bound)
# Reconstruction Loss
# KL Divergence
# Perplexity
# Latent Space Smoothness? tells whether similar emotions are clustering together in latent space (dimensional model)

import spacy


def extractclauses(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    clauses = []

    for token in doc:
        if token.dep in ("ROOT", "ccomp", "advcl", "xcomp", "relcl"):
            # Collect the span of tokens belonging to this clause
            clause_tokens = [t for t in token.subtree
                           if t.dep not in ("ccomp", "advcl", "xcomp", "relcl")
                           or t == token]

            clause_text = " ".join(t.text for t in sorted(clause_tokens, key=lambda t: t.i))
            clauses.append(clause_text)

    return clauses
