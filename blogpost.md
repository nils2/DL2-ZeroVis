# Introduction


## Related work


## Exposition

The paper delineates several key characteristics of the proposed model. First, it effectively leverages the capabilities of existing language models, including in-context learning and free-form text generation. Which allows the model to capitalize on the comprehensive knowledge encapsulated in these language models, thereby enhancing both the efficacy and efficiency of the system. Second, the model exhibits adaptness in managing cross-modal interactions, facilitating the processing of arbitrarily interleaved image and text inputs to generate coherent free-form text integrated with retrieved images. This versatility endows the model with indispensable multimodal dialogue capabilities, suitable for real-world scenarios requiring diverse input types. Finally, the model demonstrates robust zero-shot performance on grounded tasks like contextual image retrieval and multimodal dialogue, indicating a potent ability to generalize from learned concepts to new tasks without explicit task-specific training.

Such attributes inspire intriguing inquiries about the model's performance in the visual domain. Specifically, can the model effectively execute zero-shot visual arithmetic? Does the inherent in-context learning capability of large language models facilitate this task? How does the performance alter when engaged in multimodal arithmetic operations? These questions gain prominence given its design, which retrieves rather than generates images from the Conceptual Captions dataset. This makes it vulnerable to biases in the training and retrieval datasets which could potentially impact performance and results.

Our novel contributions include:

- Building upon the foundational paper that introduced FROMAGe, by offering novel insights into its visual arithmetic capabilities. Our findings will illustrate whether the model can successfully execute complex visual arithmetic operations, thereby broadening our comprehension of FROMAGe's functionality and potential applications.

- Evaluating the impact of latent few-shot in-context learning abilities of large language models (LLMs) on visual arithmetic. By investigating Chain-of-Thought reasoning on a task and modality that these LLMs are not trained on, we present the in-context abilities from a unique viewpoint divergent from previous literature. Our research discloses how the model effectively generalizes from limited examples, markedly enhancing the efficiency and precision of visual arithmetic operations.

- Demonstarting the influence of multimodal inputs on visual arithmetic. We furnish a deeper understanding of the interaction between different modalities in multimodal models, especially in tasks they are not trained on. The insights derived from this exploration bear significant implications for how multimodal models should be trained and utilized.

# Results



# Conclude

[comment]: <> (Technical limitations specific to FROMAGe:)
[comment]: <> (- does not always generate \[RET\] during inference)
[comment]: <> (- strong bias to produce regular text tokens)
[comment]: <> (Which are likely dou to its comprehensive pre-training regime (on text-only data).)
[comment]: <> (Somewhat alleviated by specifically prompting the model to ask it to show images.)

## Contributions



# References
