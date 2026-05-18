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

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import matplotlib as pyplot
import random
import matplotlib.pyplot as plt
import numpy as np
import spacy



def extractclauses(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    clauses = []

    for token in doc:
        print(token.text, token.dep_, token.head.text)
        if token.dep in ("ROOT", "ccomp", "advcl", "xcomp", "relcl"):
            print("yes i do i run :)")
            # Collect the span of tokens belonging to this clause
            clause_tokens = [t for t in token.subtree
                           if t.dep not in ("ccomp", "advcl", "xcomp", "relcl")
                           or t == token]

            # print(clause_tokens)
            clause_text = " ".join(t.text for t in sorted(clause_tokens, key=lambda t: t.i))
            clauses.append(clause_text)

    return clauses

print(extractclauses("I like to eat."))

# text = "I don't like ice cream because it's too cold."
# nlp = spacy.load('en_core_web_sm')

# See each token's grammatical role
# for token in doc:
    # print(f"{token.text:<15} dep={token.dep_:<12} head={token.head.text}")

#
# for token in doc:
#     if token.dep_ == 'ROOT':  # main verb of a clause
#         subject = [w for w in token.lefts if w.dep_ in ('nsubj', 'nsubjpass')]
#         obj     = [w for w in token.rights if w.dep_ in ('dobj', 'pobj', 'attr')]
#         print(f"verb={token.text}, subject={subject}, object={obj}")
#
# # Clause boundary dependencies to look for
# clause_deps = {'advcl', 'relcl', 'ccomp', 'xcomp', 'acl'}
#
# for token in doc:
#     if token.dep_ in clause_deps:
#         # Get the full span of this clause
#         clause_tokens = list(token.subtree)
#         clause = ' '.join(t.text for t in clause_tokens)
#         print(f"[{token.dep_}] {clause}")



# with open('../example matrixes/zA.JSON', 'r') as file:
#     data = json.load(file)
# # print(data)
# print(data["Entries"])
# # Your points (x, y, z)
# x = []
# y = []
# z = []
#
# for item in data["Entries"].values():
#     x.append(item['importance'])
#     y.append(item['valence'])
#     z.append(item['arousal'])
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(x, y, z)
#
#
# ax.set_xlabel('Importance')
# ax.set_ylabel('Valence')
# ax.set_zlabel('Arousal')
#
# ax.set_xlim(0, 1)      # importance: 0 to 1
# ax.set_ylim(-1, 1)     # valence: can go negative
# ax.set_zlim(0, 1)      # arousal: 0 to 1
#
# # Keep panes fixed so they don't flip while rotating
# ax.xaxis.pane.set_visible(True)
# ax.yaxis.pane.set_visible(True)
# ax.zaxis.pane.set_visible(True)
#
# # Lock their position by overriding the internal _PLANES variable
# ax.xaxis._axinfo['juggled'] = (0, 0, 0)
# ax.yaxis._axinfo['juggled'] = (1, 1, 1)
# ax.zaxis._axinfo['juggled'] = (2, 2, 2)
#
# ax.set_proj_type('ortho')
#
#
# plt.title('IVA Plots')
# plt.show()