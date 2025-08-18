# Augmentation methods
Two augmentation methods are explored: the argumentative cross_over method, in folder crossover_method, and the prompting method, in folder prompting_method.


# Crossover method
2 approaches are explored: Same Adverb and Louvain
Follows the following steps:
- construction of an adverbial resource containing argumentative adverbs
- identification of argumentative components using the adverbs from the resource
- recombination of argumentative components to create new samples

### adverbs.py
Contains all functions which serve to obtain the adverbial resource in both approaches.

### split.py
Contains all functions which perform the splitting into argumentative components in both approaches.

### crossover_generate.py
Contains all functions which perform the creation of new samples for recombinations.
generate_sa():  generates an augmented dataset using the Same Adverb approach. Returns it in the form of a pandas DataFrame.
generate_louvain(): generates an augmented dataset using the Same Adverb approach. Returns it in the form of a pandas DataFrame.
 
### mistral_paraphrase.py
Implementation of a paraphrasing pipeline, which given a pandas DataFrame returns its content paraphrased by the Mistral7b model, using prompting.


# Prompting method

### prompt_generation.py
Allows for the generation of YouTube comments in batches of 5 by prompting Mistral7b, with a zeroshot approach.

### **Tips**
Call generate_sa() or generate_louvain() from crossover_generate.py to augment using the crossover method.
Call zero_shot() from prompt_generation.py to augment using the prompting method.
Make sure to adapt the data and output paths.