Names: Rohan Sykora, Emily You, Martin Zuniga, Angel Li, Yusuf Abdelnur

Foxes and hedgehogs:
Is research specialization predictive of paper acceptance?

Abstract
	As a prestigious conference, an acceptance to the International Conference on Learning Representations (ICLR) defines a major milestone in a machine learning researcher’s journey. As researchers establish their presence in the ICLR space, they can choose to either continue specializing in a given area or diverge from their earlier research. While existing literature has explored the pivot penalty, the declining impact of new research as researchers shift from their previous work, within the broader nature of science, less research has examined how changes in research relates to outcomes in competitive academic conferences like the ICLR. This paper aims to investigate the relationship between research specialization and ICLR acceptance, proposing a set of methodologies. These results will assist researchers with making more informed decisions of how to approach research, whether to continue specializing or exploring a new area.

Research Question 
Given the significant time, effort, and funding that goes into publishing a research paper, and the congestion of the current machine learning research landscape, we are interested in examining how the specialization of researchers in the ICLR affects paper acceptance. Our methodology can be organized into three main parts: the embeddings/representations of the abstract, our specialization (“hedgehog”) score, and the prediction classifier. 

Dataset and Preprocessing
Our main dataset is the ICLR OpenReview Dataset constructed by Marquez-Gonzalez and Kobak (2024). The dataset contains each paper’s year, title, abstract, authors, the decision (accept or reject), paper review scores, keywords, and labels. During preprocessing, we first grouped papers by principal investigator (PI), which is the last listed author in a paper. More in-depth exploration of the dataset can be seen in our Jupyter notebook.

Methods
Paper representation. We embed each submission’s title and abstract using a model(s?) built on SciBERT. The original paper uses scientific specialized domain models DeCLUTR-sci (Giorgi et al., 2021), SciNCL (Ostendorff et al., 2022) and SPECTER (Cohan et al., 2020). The original paper also uses tf-IDF to create a bag-of-words embedding for each article. Using these approaches to creating our representations will yield vectors in a high-dimensional space that we can use to make our comparisons between different papers. 

Quantifying specialization. 
Specialization as inverse area spanned by papers:  We represent each author’s research history as a cloud of points in embedding space: each point is one of their papers, placed by the SPECTER embedding of its title and abstract. Because papers on unrelated topics land far apart, the shape of the cloud – defined as the convex hull enclosed by the author’s papers – describes the author’s research spread. By extension, the inverse of this enclosed area represents specialization. Since computing the convex hull area (CHA) in a high-dimensional space is computationally intensive and sensitive to near-identical papers, we extract the top components explaining at least 90% of the variance and compute the CHA in this lower-dimensional space (Ozgur et al. 2024). Importantly, this measure needs to be adjusted by the number of papers to make volumes comparable across authors use a fixed window m so that volumes are comparable across authors.

Pivot index: Similar to Hill et al. (2025), we plan on quantifying how different a new paper is compared to the author’s prior work. We will compute this by using the cosine similarity between (1) a new paper’s embedding vector representation and (2) an aggregate embedding vector representing all prior work in the H(a,t) set. Mathematically, the pivot index of prior work j and current work i of author a is computed as
ji = 1 -eH(a,t)ei|eH(a,t)||ei| 
where eH(a,t) is the vector representing the mean-pooled BERT topic distribution for author a at time t (time is measured in publication steps e.g., time 2 is the time of paper 2 publication). ei  is the BERT representation of the new paper. If the new paper is similar in topic embeddings to the prior work, their pivot index will be closer to 0.

Topic entropy: Using BERTopic, we extract a set of labels from each paper, which will be used to construct an author’s profile based on the frequency of these labels. We would then implement Shannon Entropy,  S(X) = −∑ p(xi) log2(p(xi)), with p(xi) being the probability of having paper coverage in a particular area x. Higher entropy would be indicative of more dispersed information content and thus a lack of specialization. As with method (1), this measure also needs to be adjusted by the number of papers.

Divergence from general population: Specialization can be evaluated on the basis of whether or not particular authors have deviated from their peers. With an established set of years for testing, we will identify a group of authors of similar topic specialty (e.g., authors who work predominantly in Deep Learning). Consequently, we can examine how each author deviates from peers within the similar-topic group and determine whether divergence contributes to paper acceptance. 

Proposed Timeline

Week of 11/1 - 11/7: All group effort
Conduct exploratory analysis – group papers by Principal Investigator, identify and remove empty and "anonymous" authors, plot minimum papers threshold (k) against % of authors retained. 
Week of 11/8 - 11/14: Angel, Yusuf take lead, group supports as needed
Create representations of the different papers using the models described in the methods section, and do some exploratory analysis of the space of these different types of embeddings and create visualizations using BERTopic. 
Week of 11/15 - 11/21: Martin, Emily take lead, group supports as needed
Begin to group papers at the author level and experiment with the different specialization quantification metrics. By the end of this week or early next week, our definition of specialization will be concrete.  
Week of 11/22 - 11/28: All group effort
Thanksgiving break, account for confounding metrics such as gender, number of collaborators, total years publishing, etc. 
Week of 11/29 - 12/5: Rohan take lead, group supports as needed
Fit a linear regression model, where given a specialization profile and the confounding variables, we see how much specialization impacts the probability of the next paper being accepted. Also conduct general correlation analysis between specialization and papers accepted and scores, controlling for the numbers of papers submitted. 
Week of 12/6 - 12/12: Rohan take lead, group supports as needed
 Begin to write the results of our study and move forward. 
Week of 12/13 - 12/17: All group effort
Finish writing our paper and submit!
References 
Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A pretrained language model for scientific text. arXiv. https://doi.org/10.48550/arXiv.1903.10676
Catak, F. O., Kuzlu, M., & Patrick, T. (2024). Improving medical diagnostics with vision-language models: Convex hull-based uncertainty analysis. arXiv. https://doi.org/10.48550/arXiv.2412.00056
Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. (2020). SPECTER: Document-level representation learning using citation-informed transformers. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2270–2282. https://doi.org/10.18653/v1/2020.acl-main.207
González-Márquez, R., & Kobak, D. (2024). Learning representations of learning representations. arXiv preprint arXiv:2404.08403.
Giorgi, J., Nitski, O., Wang, B., & Bader, G. (2021). DeCLUTR: Deep contrastive learning for unsupervised textual representations. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 879–895. https://doi.org/10.18653/v1/2021.acl-long.72
Hill, R., Yin, Y., Stein, C., Wang, X., Wang, D., & Jones, B. F. (2025). The pivot penalty in research. Nature, 642(8069), 999-1006.
"ICLR 2027 Author Guidelines." International Conference on Learning Representations, 2026,     iclr.cc/Conferences/2027/AuthorGuidelines. 
Ostendorff, M., Rethmeier, N., Augenstein, I., Gipp, B., & Rehm, G. (2022). Neighborhood contrastive learning for scientific document representations with citation embeddings. Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 11670–11688. https://doi.org/10.18653/v1/2022.emnlp-main.802