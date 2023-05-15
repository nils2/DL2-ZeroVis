# Chain-of-Thought Prompting and Visual Arithmetic Tasks: Exploring Grounding in FROMAGe
> Joy Crosbie, Emile Dhifallah, Dominik Martínez, Gergely Papp, Nils Peters

## Introduction

### FROMAGe

In the era of evergrowing language models, a key challenge lies in the lack of multimodal grounding. Language models, despite being trained on massive text corpora and demonstrating impressive capabilities such as generating human-like dialogue and answering complex questions, are generally incapable of incorporating visual cues. This significantly limits their performance on tasks that require visual reasoning and grounding. The paper, "Grounding Language Models to Images for Multimodal Generation", addresses this limitation by introducing the FROMAGe (Frozen Retrieval Over Multimodal Data for Autoregressive Generation) model, which extends the capabilities of a standard text-only language model to a multimodal (text and image) realm.

FROMAGe (Koh et al., 2023) employs a pretrained language model and a visual encoder, both kept frozen, to ground language models visually. This grounding is achieved by training the model with a multitask objective for image captioning and image-text retrieval. For image captioning, the model learns to extract visual embeddings and map them into the input space of the language model through a maximum likelihood objective. On the other hand, for image-text retrieval, the model uses contrastive learning to map the embeddings of a new [RET] token, which represents an image, to the corresponding visual embeddings. This process only updates the weights of the linear layers and the [RET] token embedding, ensuring computational and memory efficiency.

The paper by Koh et al. (2023) highlights several key strengths of the proposed model. Firstly, FROMAGe retains the original abilities of the text-only language model, such as text generation, while also acquiring new multimodal dialogue and reasoning abilities. Secondly, the model demonstrates strong few-shot multimodal abilities learned from image-caption pairs alone, which is a standout feature as most models require web-scale interleaved image-text data (Alayrac et al., 2022; Aghajanyan et al., 2022). Thirdly, it demonstrates superior text-to-image retrieval sensitivity, especially for long and complex free-form text. Lastly, FROMAGe enhances the existing capabilities of pretrained language models, such as in-context learning, input sensitivity, and dialogue generation, for visually grounded tasks. 



### Visual Arithmetic Tasks and Grounding

In the context of deep learning, visual arithmetic tasks are not about traditional numerical operations, but rather about finding analogies or relationships between different entities. This concept takes inspiration from the vector arithmetic introduced by Mikolov et al. (2013). In their work, they demonstrated that word embeddings can be manipulated using arithmetic operations to reveal semantic relationships between words. Extending this idea into the visual domain, visual arithmetic tasks involve understanding and applying a transformation observed in a pair of images to another pair. A classic example would be: if "Obama" is to "USA" as "X" is to "Germany", what is "X"? The correct answer would ideally be "Angela Merkel", following the logic of leaders to their respective countries.

Building on this concept, Tewel et al. (2022) in their work on ZeroCap, applied arithmetic operations to visual embeddings to "generate knowledge of the external world". They perform calculations over a combination of visual and textual inputs by operating in the latent embedding space. However, their methodology relies on CLIP (Radford et al., 2021), a model that can handle visual and textual inputs but only generates textual outputs, thus limiting their arithmetic to natural language output:

> "Who is the president of Germany?"\
> X = image(Obama) - image(USA) + image(Germany) &rarr; "Angela Merkel"

Visual arithmetic tasks provide an interesting paradigm for exploring the grounding capabilities of multimodal language models. These tasks involve the ability to comprehend, reason, and make decisions based on visual inputs - a challenge that requires a deep level of grounding in the visual world. By applying the FROMAGe model to these tasks, we can gain a better understanding of the extent to which it is truly "grounded". Moreover, with FROMAGe's ability to process and output multimodal data, we are equipped to extend the arithmetic operations’ output into the visual domain, overcoming the limitations faced by models such as CLIP.



## This section needs a title

### Visual Relations Benchmark
We make use of the Visual Relations Benchmark introduced in the ZeroCap (Tewel et al., 2023) paper. This benchmark encompasses 320 distinct relationships distributed among image templates, such as buildings→country, countries→capital, foods→country, leaders→country, and CEO→company. These relations were specifically chosen for their many-to-one association, exemplified by the fact that a country can host a myriad of buildings, yet each building typically pertains to a single country. The benchmark is devised to gauge two primary capabilities: the modeling of visual relations and the application of worldly knowledge in task execution. Although originally devised for single-word answer generation, this dataset also facilitates the retrieval of images that correctly demonstrate visual arithmetic. It is therefore able to handle the multi-modal arithmetics that will be performed using FROMAGe.


### Applying FROMAGe to Visual Arithmetic Tasks

Applying the FROMAGe model to visual arithmetic tasks involves processing each image with the visual encoder to extract visual embeddings. These embeddings are mapped, using the pretrained linear mapping of the model, into the input space of the language model. The newly mapped text-like query describing the analogy task is interleaved with the [RET] token.

The FROMAGe model is then tasked to generate a response that correctly completes the analogy. The model's grounding in both language and visual inputs should ideally enable it to understand the relationship between the entities and provide a coherent, accurate response - for example, returning an image of Angela Merkel in response to the analogy task involving images of Obama, the USA, and Germany.

EXAMPLE OF PROMPT WILL BE SHOWN HERE, FOR NOW:
> "Who is the president of Germany?"\
> X = image(Obama) - image(USA) + image(Germany) &rarr; image(Angela Merkel)


CORRESPONDING RESULTS WILL BE SHOWN HERE

#### Chain-of-Thought Prompting for Visual Arithmetic
Chain-of-thought prompting is a technique aimed at enhancing the reasoning ability of large language models. Rather than presenting a prompt in isolation, it involves including a series of intermediate reasoning steps in natural language within the few-shot prompting process (Kojima et al. 2022). This has been shown to improve performance, particularly for complex reasoning tasks (Wei et al. 2022, Suzgun et al. 2022). When applied in combination with visual arithmetic tasks, it can offer deeper insights into how well the model understands and connects visual and linguistic cues.

In the context of visual arithmetic tasks, chain-of-thought prompting would involve presenting the FROMAGe model with a set of visual analogies along with a series of intermediate reasoning steps. For instance, instead of directly asking "Obama is to the USA as X is to Germany", the model would be guided through the reasoning process: 

EXAMPLE OF PROMPT WILL BE SHOWN HERE

CORRESPONDING RESULTS WILL BE SHOWN HERE

#### T-SNE
In this study, we leverage t-Distributed Stochastic Neighbor Embedding (T-SNE) (Van der Maaten et al., 2008), a non-linear dimensionality reduction technique that is particularly adept at preserving local structure within high-dimensional datasets. T-SNE calculates the similarity of data points in the high-dimensional space and then maps it to a lower-dimensional space. It uses gradient descent to minimizes the Kullback-Leibler (KL) divergence between the high and low-dimensional representations with respect to the locations of the points in the map. The output is a two- or three-dimensional representation of the data that can be easily visualized, preserving the structure and relationships inherent in the high-dimensional data space as much as possible. This dimensionality reduction algorithm is used to visualize nonlinear relations between the image embeddings, allowing for an better analysis of the retrieved tokens from the FROMAGe model.

## Conclusion
## Novel Contributions
Our novel contributions include:
* Building upon the foundational paper that introduced FROMAGe, by offering novel insights into its visual arithmetic capabilities. Our findings will illustrate whether the model can successfully execute complex visual arithmetic operations, thereby broadening our comprehension of FROMAGe's functionality and potential applications.
* Evaluating the impact of latent few-shot in-context learning abilities of large language models (LLMs) on visual arithmetic. By investigating Chain-of-Thought reasoning on a task and modality that these LLMs are not trained on, we present the in-context abilities from a unique viewpoint divergent from previous literature. Our research discloses how the model effectively generalizes from limited examples, markedly enhancing the efficiency and precision of visual arithmetic operations.
* Demonstrating the influence of multimodal inputs on visual arithmetic. We furnish a deeper understanding of the interaction between different modalities in multimodal models, especially in tasks they are not trained on. The insights derived from this exploration bear significant implications for how multimodal models should be trained and utilized.

## Individual Contributions

