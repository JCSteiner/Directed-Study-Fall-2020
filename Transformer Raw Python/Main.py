#%%########################## IMPORT DEPENDENCIES #############################

#Imports the ability to work with advanced matrix functions
import numpy as np

# Imports our transformer model. Transformers are sequence to sequence
# operations that take in a sequence and output a sequence in return
from Transformer import Transformer

#%%############################ DEFINE CONSTANTS ##############################

# Our first constant is EMBEDDING_DIMS this is how many dimensions we transform
# our inputs into
EMBEDDING_DIMS = 2

# Our next constant is VOCAB_SIZE, this is how many possible inputs we could
# have, and how many outputs we will transform into
VOCAB_SIZE = 4

# Our third constant is NUM_HEADS, this is how many heads (concurrent sets of
# weights) our self-attention mechanism will have
NUM_HEADS = 3

# Our fourth constant is NUM_BLOCKS, this is how many transformer blocks we
# will have. This will help us calculate more complex relationships between
# sets of sequences just like multiple layers would in any other type of
# neural network.
NUM_BLOCKS = 2

#%%#######################
np.random.seed(1)
x = [0, 1]
targets = 3
model = Transformer(EMBEDDING_DIMS, VOCAB_SIZE, NUM_HEADS, NUM_BLOCKS)
pred = model.fwdPass(x)
print("Predicted:", np.argmax(pred), "Actual:", targets)
print("Training...")
for _ in range(10):
    model.zeroGrad()
    cost = 0
    pred = model.fwdPass(x)
    cost += model.bkwdPass(pred, targets)
    model.step(0.01, 1)
    print("Cost:", cost)
pred = model.fwdPass(x)
print("Predicted:", np.argmax(pred), "Actual:", targets)